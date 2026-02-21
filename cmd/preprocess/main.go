package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/klauspost/compress/zstd"
	"github.com/notnil/chess"
	"github.com/xitongsys/parquet-go-source/local"
	"github.com/xitongsys/parquet-go/source"
	"github.com/xitongsys/parquet-go/writer"
)

const shardSize = 100000
const progressInterval = 5 * time.Second

const (
	nullID int32 = 0
)

const (
	playerWhite int32 = 1
	playerBlack int32 = 2
)

type gameFeatures struct {
	Piece         []int32 `parquet:"name=piece, type=INT32, repetitiontype=REPEATED"`
	From          []int32 `parquet:"name=from_square, type=INT32, repetitiontype=REPEATED"`
	To            []int32 `parquet:"name=to_square, type=INT32, repetitiontype=REPEATED"`
	Capture       []int32 `parquet:"name=capture, type=INT32, repetitiontype=REPEATED"`
	Promotion     []int32 `parquet:"name=promotion, type=INT32, repetitiontype=REPEATED"`
	Check         []int32 `parquet:"name=check, type=INT32, repetitiontype=REPEATED"`
	Castle        []int32 `parquet:"name=castle, type=INT32, repetitiontype=REPEATED"`
	Player        []int32 `parquet:"name=player, type=INT32, repetitiontype=REPEATED"`
	UCIMove       []int32 `parquet:"name=uci_move, type=INT32, repetitiontype=REPEATED"`
	CompositeMove []int32 `parquet:"name=composite_move, type=INT32, repetitiontype=REPEATED"`
}

type pgnJob struct {
	moves  string
	winner chess.Color
}

type progressUpdate struct {
	decoded uint64
	skipped uint64
	written uint64
	shards  uint64
}

var sanDecoder = chess.AlgebraicNotation{}
var longDecoder = chess.LongAlgebraicNotation{}
var uciDecoder = chess.UCINotation{}

// UCI vocab: move string → 1-indexed ID (0 = null/padding, consistent with other features)
var uciVocab map[string]int32

// Composite vocab: 8-feature tuple → 1-indexed ID
type compositeTuple [8]int32

var compositeVocab map[compositeTuple]int32

func loadCompositeVocab(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	compositeVocab = make(map[compositeTuple]int32)
	scanner := bufio.NewScanner(f)
	var id int32
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Split(line, "\t")
		if len(parts) != 8 {
			continue
		}
		var tup compositeTuple
		for i, p := range parts {
			val, _ := strconv.Atoi(p)
			tup[i] = int32(val)
		}
		compositeVocab[tup] = id + 1 // 1-indexed
		id++
	}
	return scanner.Err()
}

func compositeMoveID(piece, fromSq, toSq, capture, promotion, check, castle, player int32) int32 {
	if compositeVocab == nil {
		return nullID
	}
	tup := compositeTuple{piece, fromSq, toSq, capture, promotion, check, castle, player}
	if id, ok := compositeVocab[tup]; ok {
		return id
	}
	return nullID
}

func loadUCIVocab(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	uciVocab = make(map[string]int32)
	scanner := bufio.NewScanner(f)
	var id int32
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			uciVocab[line] = id + 1 // 1-indexed
			id++
		}
	}
	return scanner.Err()
}

func uciStr(from, to int, promoChar byte) string {
	s := []byte{
		byte('a' + from%8), byte('1' + from/8),
		byte('a' + to%8), byte('1' + to/8),
	}
	if promoChar != 0 {
		s = append(s, promoChar)
	}
	return string(s)
}

func litePromoChar(p int8) byte {
	switch p {
	case liteKnight:
		return 'n'
	case liteBishop:
		return 'b'
	case liteRook:
		return 'r'
	case liteQueen:
		return 'q'
	}
	return 0
}

func chessPromoChar(pt chess.PieceType) byte {
	switch pt {
	case chess.Knight:
		return 'n'
	case chess.Bishop:
		return 'b'
	case chess.Rook:
		return 'r'
	case chess.Queen:
		return 'q'
	}
	return 0
}

func uciMoveID(from, to int, promoChar byte) int32 {
	if uciVocab == nil {
		return nullID
	}
	if id, ok := uciVocab[uciStr(from, to, promoChar)]; ok {
		return id
	}
	return nullID
}

type zstdReadCloser struct {
	dec  *zstd.Decoder
	file *os.File
}

func (z *zstdReadCloser) Read(p []byte) (int, error) {
	return z.dec.Read(p)
}

func (z *zstdReadCloser) Close() error {
	z.dec.Close()
	return z.file.Close()
}

func openPGN(path string) (io.ReadCloser, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	if filepath.Ext(path) == ".zst" {
		decoder, err := zstd.NewReader(file)
		if err != nil {
			file.Close()
			return nil, err
		}
		return &zstdReadCloser{dec: decoder, file: file}, nil
	}
	return file, nil
}

func squareID(s chess.Square) int32 {
	// Map a1..h8 to 1..64, rank-major.
	if s == chess.NoSquare {
		return nullID
	}
	file := s.File()
	rank := s.Rank()
	return int32(1 + int(rank)*8 + int(file))
}

func pieceID(p chess.Piece) int32 {
	switch p.Type() {
	case chess.Pawn:
		return 1
	case chess.Knight:
		return 2
	case chess.Bishop:
		return 3
	case chess.Rook:
		return 4
	case chess.Queen:
		return 5
	case chess.King:
		return 6
	default:
		return nullID
	}
}

func playerID(turn chess.Color) int32 {
	if turn == chess.White {
		return playerWhite
	}
	return playerBlack
}

func promotionID(p chess.PieceType) int32 {
	switch p {
	case chess.Knight:
		return 1
	case chess.Bishop:
		return 2
	case chess.Rook:
		return 3
	case chess.Queen:
		return 4
	default:
		return nullID
	}
}

func castleID(m *chess.Move) int32 {
	if m.HasTag(chess.KingSideCastle) {
		return 1
	}
	if m.HasTag(chess.QueenSideCastle) {
		return 2
	}
	return nullID
}

func captureID(m *chess.Move) int32 {
	if m.HasTag(chess.Capture) {
		return 1
	}
	return nullID
}

func pieceIDFromToken(moveStr string, pos *chess.Position, m *chess.Move) int32 {
	if len(moveStr) > 0 {
		switch moveStr[0] {
		case 'K':
			return 6
		case 'Q':
			return 5
		case 'R':
			return 4
		case 'B':
			return 3
		case 'N':
			return 2
		case 'O':
			return 6
		default:
			return 1
		}
	}
	return pieceID(pos.Board().Piece(m.S1()))
}

func checkIDFromToken(moveStr string, m *chess.Move) int32 {
	if len(moveStr) > 0 {
		last := moveStr[len(moveStr)-1]
		if last == '#' {
			return 2
		}
		if last == '+' {
			return 1
		}
	}
	if m.HasTag(chess.Check) {
		return 1
	}
	return nullID
}

func isSpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

func isMoveNumberToken(token string) bool {
	if token == "" {
		return false
	}
	seenDigit := false
	seenDot := false
	for i := 0; i < len(token); i++ {
		ch := token[i]
		if ch >= '0' && ch <= '9' {
			seenDigit = true
			continue
		}
		if ch == '.' {
			seenDot = true
			continue
		}
		return false
	}
	return seenDigit && seenDot
}

func stripMoveSuffix(token string) string {
	end := len(token)
	for end > 0 {
		last := token[end-1]
		if last != '!' && last != '?' {
			break
		}
		end--
	}
	return token[:end]
}

func nextToken(movetext string, i int) (string, int) {
	n := len(movetext)
	for i < n && isSpace(movetext[i]) {
		i++
	}
	start := i
	for i < n {
		ch := movetext[i]
		if isSpace(ch) || ch == '{' || ch == '}' || ch == '(' || ch == ')' {
			break
		}
		i++
	}
	if i == start {
		return "", i
	}
	return movetext[start:i], i
}

func decodeMove(pos *chess.Position, moveStr string) (*chess.Move, error) {
	move, err := sanDecoder.Decode(pos, moveStr)
	if err == nil {
		return move, nil
	}
	move, err = longDecoder.Decode(pos, moveStr)
	if err == nil {
		return move, nil
	}
	return uciDecoder.Decode(pos, moveStr)
}

func buildFeaturesFromMoves(movetext string, winner chess.Color) (*gameFeatures, bool) {
	pos := chess.StartingPosition()
	capHint := strings.Count(movetext, " ")/2 + 4
	if capHint < 8 {
		capHint = 8
	}
	features := &gameFeatures{
		Piece:         make([]int32, 0, capHint),
		From:          make([]int32, 0, capHint),
		To:            make([]int32, 0, capHint),
		Capture:       make([]int32, 0, capHint),
		Promotion:     make([]int32, 0, capHint),
		Check:         make([]int32, 0, capHint),
		Castle:        make([]int32, 0, capHint),
		Player:        make([]int32, 0, capHint),
		UCIMove:       make([]int32, 0, capHint),
		CompositeMove: make([]int32, 0, capHint),
	}

	inVariation := 0
	i := 0
	for i < len(movetext) {
		ch := movetext[i]
		if ch == '{' {
			i++
			for i < len(movetext) && movetext[i] != '}' {
				i++
			}
			if i < len(movetext) {
				i++
			}
			continue
		}
		if ch == '(' {
			inVariation++
			i++
			continue
		}
		if ch == ')' {
			if inVariation > 0 {
				inVariation--
			}
			i++
			continue
		}
		if inVariation > 0 || isSpace(ch) {
			i++
			continue
		}

		token, nextIdx := nextToken(movetext, i)
		if nextIdx == i {
			i++
			continue
		}
		i = nextIdx
		if token == "" {
			continue
		}
		if token[0] == '$' {
			continue
		}
		if token == "1-0" || token == "0-1" || token == "1/2-1/2" || token == "*" {
			break
		}
		if isMoveNumberToken(token) {
			continue
		}

		moveStr := stripMoveSuffix(token)
		if moveStr == "" {
			continue
		}

		move, err := decodeMove(pos, moveStr)
		if err != nil {
			return nil, false
		}
		nextPos := pos.Update(move)

		pID := pieceIDFromToken(moveStr, pos, move)
		fID := squareID(move.S1())
		tID := squareID(move.S2())
		capID := captureID(move)
		promoID := promotionID(move.Promo())
		chkID := checkIDFromToken(moveStr, move)
		castID := castleID(move)
		plID := playerID(pos.Turn())

		features.Piece = append(features.Piece, pID)
		features.From = append(features.From, fID)
		features.To = append(features.To, tID)
		features.Capture = append(features.Capture, capID)
		features.Promotion = append(features.Promotion, promoID)
		features.Check = append(features.Check, chkID)
		features.Castle = append(features.Castle, castID)
		features.Player = append(features.Player, plID)
		features.UCIMove = append(features.UCIMove, uciMoveID(
			int(move.S1()), int(move.S2()), chessPromoChar(move.Promo()),
		))
		features.CompositeMove = append(features.CompositeMove, compositeMoveID(
			pID, fID, tID, capID, promoID, chkID, castID, plID,
		))

		pos = nextPos
	}

	if len(features.Piece) == 0 {
		return nil, false
	}

	return features, true
}

func parseResultTag(line string) (chess.Color, bool) {
	if !strings.HasPrefix(line, "[Result ") {
		return chess.NoColor, false
	}
	if strings.Contains(line, "\"1-0\"") {
		return chess.White, true
	}
	if strings.Contains(line, "\"0-1\"") {
		return chess.Black, true
	}
	return chess.NoColor, false
}

func streamJobs(reader io.Reader, jobs chan<- pgnJob, maxGames int) error {
	scan := bufio.NewScanner(bufio.NewReaderSize(reader, 4*1024*1024))
	scan.Buffer(make([]byte, 0, 1024*1024), 16*1024*1024)

	var b strings.Builder
	sawTag := false
	inMoves := false
	winner := chess.NoColor
	hasWinner := false

	count := 0
	flush := func() bool {
		if sawTag && inMoves && hasWinner {
			jobs <- pgnJob{moves: b.String(), winner: winner}
			count++
			if maxGames > 0 && count >= maxGames {
				return true
			}
		}
		b.Reset()
		sawTag = false
		inMoves = false
		winner = chess.NoColor
		hasWinner = false
		return false
	}

	for scan.Scan() {
		raw := scan.Text()
		line := strings.TrimSpace(raw)
		if line == "" {
			if inMoves {
				if flush() {
					return nil
				}
			}
			continue
		}
		if strings.HasPrefix(line, "[") {
			if inMoves {
				if flush() {
					return nil
				}
			}
			sawTag = true
			if w, ok := parseResultTag(line); ok {
				winner = w
				hasWinner = true
			}
			continue
		}
		if sawTag {
			inMoves = true
			b.WriteString(raw)
			b.WriteByte('\n')
		}
	}
	if sawTag && inMoves {
		_ = flush()
	}
	return scan.Err()
}

type shardWriter struct {
	fw source.ParquetFile
	pw *writer.ParquetWriter
}

func newShardWriter(outputDir string, shardIdx int) (*shardWriter, error) {
	filePath := filepath.Join(outputDir, fmt.Sprintf("shard-%05d.parquet", shardIdx))
	fw, err := local.NewLocalFileWriter(filePath)
	if err != nil {
		return nil, err
	}
	pw, err := writer.NewParquetWriter(fw, new(gameFeatures), int64(runtime.NumCPU()))
	if err != nil {
		fw.Close()
		return nil, err
	}
	pw.RowGroupSize = 128 * 1024 * 1024
	pw.PageSize = 8 * 1024
	return &shardWriter{fw: fw, pw: pw}, nil
}

func (s *shardWriter) Close() error {
	if err := s.pw.WriteStop(); err != nil {
		s.fw.Close()
		return err
	}
	return s.fw.Close()
}

func main() {
	var pgnPath string
	var outDir string
	var maxGames int
	var vocabPath string
	var compositeVocabPath string
	flag.StringVar(&pgnPath, "pgn", "", "path to PGN (.pgn or .pgn.zst)")
	flag.StringVar(&outDir, "out", "", "output directory for parquet shards")
	flag.IntVar(&maxGames, "max-games", 0, "optional cap on processed decisive games")
	flag.StringVar(&vocabPath, "vocab", "uci_vocab.txt", "path to UCI move vocab (from vocabbuilder)")
	flag.StringVar(&compositeVocabPath, "composite-vocab", "composite_vocab.txt", "path to composite vocab (from compositevocab)")
	flag.Parse()

	if pgnPath == "" || outDir == "" {
		fmt.Println("usage: preprocess -pgn <path> -out <dir>")
		os.Exit(2)
	}

	if err := loadUCIVocab(vocabPath); err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not load UCI vocab from %s: %v (uci_move column will be zeros)\n", vocabPath, err)
	} else {
		fmt.Fprintf(os.Stderr, "loaded UCI vocab: %d moves\n", len(uciVocab))
	}

	if err := loadCompositeVocab(compositeVocabPath); err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not load composite vocab from %s: %v (composite_move column will be zeros)\n", compositeVocabPath, err)
	} else {
		fmt.Fprintf(os.Stderr, "loaded composite vocab: %d tuples\n", len(compositeVocab))
	}

	if err := os.MkdirAll(outDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "failed to create output dir: %v\n", err)
		os.Exit(1)
	}

	reader, err := openPGN(pgnPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to open pgn: %v\n", err)
		os.Exit(1)
	}
	defer reader.Close()

	jobs := make(chan pgnJob, runtime.NumCPU()*8)
	results := make(chan gameFeatures, runtime.NumCPU()*2)
	parseErr := make(chan error, 1)
	progressUpdates := make(chan progressUpdate, runtime.NumCPU()*4)

	progressDone := make(chan struct{})
	start := time.Now()
	go func() {
		var decoded uint64
		var skipped uint64
		var written uint64
		var shards uint64
		ticker := time.NewTicker(progressInterval)
		defer ticker.Stop()

		printStatus := func(final bool) {
			elapsed := time.Since(start).Truncate(time.Second)
			fmt.Fprintf(
				os.Stderr,
				"\rdecoded %d | written %d | skipped %d | shards %d | elapsed %s",
				decoded,
				written,
				skipped,
				shards,
				elapsed,
			)
			if final {
				fmt.Fprintln(os.Stderr)
			}
		}

		for {
			select {
			case update, ok := <-progressUpdates:
				if !ok {
					printStatus(true)
					close(progressDone)
					return
				}
				decoded += update.decoded
				skipped += update.skipped
				written += update.written
				shards += update.shards
			case <-ticker.C:
				printStatus(false)
			}
		}
	}()

	var wg sync.WaitGroup
	workerCount := runtime.NumCPU()
	for i := 0; i < workerCount; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			decodedLocal := 0
			skippedLocal := 0
			flushTicker := time.NewTicker(progressInterval)
			defer flushTicker.Stop()

			flush := func() {
				if decodedLocal == 0 && skippedLocal == 0 {
					return
				}
				progressUpdates <- progressUpdate{
					decoded: uint64(decodedLocal),
					skipped: uint64(skippedLocal),
				}
				decodedLocal = 0
				skippedLocal = 0
			}

			for {
				select {
				case job, ok := <-jobs:
					if !ok {
						flush()
						return
					}
					feats, ok := buildFeaturesFromMovesLite(job.moves, job.winner)
					if !ok {
						skippedLocal++
						continue
					}
					results <- *feats
					decodedLocal++
				case <-flushTicker.C:
					flush()
				}
			}
		}()
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	go func() {
		parseErr <- streamJobs(reader, jobs, maxGames)
		close(jobs)
	}()

	shardIdx := 0
	shardCount := 0
	writtenLocal := 0
	shardsLocal := 0
	lastWriterFlush := time.Now()
	flushWriterProgress := func(force bool) {
		if !force && time.Since(lastWriterFlush) < progressInterval {
			return
		}
		if writtenLocal == 0 && shardsLocal == 0 {
			lastWriterFlush = time.Now()
			return
		}
		progressUpdates <- progressUpdate{
			written: uint64(writtenLocal),
			shards:  uint64(shardsLocal),
		}
		writtenLocal = 0
		shardsLocal = 0
		lastWriterFlush = time.Now()
	}

	var currentShard *shardWriter
	for feats := range results {
		if currentShard == nil {
			var err error
			currentShard, err = newShardWriter(outDir, shardIdx)
			if err != nil {
				fmt.Fprintf(os.Stderr, "failed to open shard: %v\n", err)
				os.Exit(1)
			}
		}
		if err := currentShard.pw.Write(feats); err != nil {
			fmt.Fprintf(os.Stderr, "write shard error: %v\n", err)
			os.Exit(1)
		}
		writtenLocal += 1
		shardCount += 1
		if shardCount >= shardSize {
			if err := currentShard.Close(); err != nil {
				fmt.Fprintf(os.Stderr, "close shard error: %v\n", err)
				os.Exit(1)
			}
			shardCount = 0
			currentShard = nil
			shardIdx += 1
			shardsLocal += 1
		}
		flushWriterProgress(false)
	}

	if err := <-parseErr; err != nil {
		fmt.Fprintf(os.Stderr, "scanner error: %v\n", err)
		os.Exit(1)
	}

	if currentShard != nil {
		if err := currentShard.Close(); err != nil {
			fmt.Fprintf(os.Stderr, "close shard error: %v\n", err)
			os.Exit(1)
		}
		shardsLocal += 1
	}
	flushWriterProgress(true)
	close(progressUpdates)
	<-progressDone

	totalWritten := shardIdx*shardSize + shardCount
	if totalWritten == 0 {
		fmt.Println("done. games=0 shards=0")
		return
	}
	finalShards := shardIdx
	if shardCount > 0 {
		finalShards = shardIdx + 1
	}
	fmt.Printf("done. games=%d shards=%d\n", totalWritten, finalShards)
}
