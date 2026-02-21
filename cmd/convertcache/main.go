// convertcache reads old-format parquet shards (with step column and
// 4-value player) and writes new-format shards (no step, 2-value player,
// composite_move column).
//
// Player mapping:
//
//	old self-white=1  -> new white=1
//	old self-black=2  -> new black=2
//	old opponent-white=3 -> new white=1
//	old opponent-black=4 -> new black=2
//
// Usage:
//
//	convertcache -in data/cache/old -out data/cache/new -composite-vocab composite_vocab.txt
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/xitongsys/parquet-go-source/local"
	"github.com/xitongsys/parquet-go/reader"
	"github.com/xitongsys/parquet-go/writer"
)

// Old schema (reading)
type oldGameFeatures struct {
	Piece     []int32 `parquet:"name=piece, type=INT32, repetitiontype=REPEATED"`
	From      []int32 `parquet:"name=from_square, type=INT32, repetitiontype=REPEATED"`
	To        []int32 `parquet:"name=to_square, type=INT32, repetitiontype=REPEATED"`
	Capture   []int32 `parquet:"name=capture, type=INT32, repetitiontype=REPEATED"`
	Promotion []int32 `parquet:"name=promotion, type=INT32, repetitiontype=REPEATED"`
	Check     []int32 `parquet:"name=check, type=INT32, repetitiontype=REPEATED"`
	Castle    []int32 `parquet:"name=castle, type=INT32, repetitiontype=REPEATED"`
	Step      []int32 `parquet:"name=step, type=INT32, repetitiontype=REPEATED"`
	Player    []int32 `parquet:"name=player, type=INT32, repetitiontype=REPEATED"`
	UCIMove   []int32 `parquet:"name=uci_move, type=INT32, repetitiontype=REPEATED"`
}

// New schema (writing) — no Step, remapped Player, + CompositeMove
type newGameFeatures struct {
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
		return 0
	}
	tup := compositeTuple{piece, fromSq, toSq, capture, promotion, check, castle, player}
	if id, ok := compositeVocab[tup]; ok {
		return id
	}
	return 0
}

func remapPlayer(old int32) int32 {
	switch old {
	case 1: // self-white -> white
		return 1
	case 2: // self-black -> black
		return 2
	case 3: // opponent-white -> white
		return 1
	case 4: // opponent-black -> black
		return 2
	default:
		return 0
	}
}

func convertShard(inPath, outPath string) (int, error) {
	fr, err := local.NewLocalFileReader(inPath)
	if err != nil {
		return 0, fmt.Errorf("open reader: %w", err)
	}
	defer fr.Close()

	pr, err := reader.NewParquetReader(fr, new(oldGameFeatures), int64(runtime.NumCPU()))
	if err != nil {
		return 0, fmt.Errorf("init parquet reader: %w", err)
	}
	defer pr.ReadStop()

	numRows := int(pr.GetNumRows())
	if numRows == 0 {
		return 0, nil
	}

	fw, err := local.NewLocalFileWriter(outPath)
	if err != nil {
		return 0, fmt.Errorf("open writer: %w", err)
	}

	pw, err := writer.NewParquetWriter(fw, new(newGameFeatures), int64(runtime.NumCPU()))
	if err != nil {
		fw.Close()
		return 0, fmt.Errorf("init parquet writer: %w", err)
	}
	pw.RowGroupSize = 128 * 1024 * 1024
	pw.PageSize = 8 * 1024

	closePW := func() error {
		if err := pw.WriteStop(); err != nil {
			fw.Close()
			return err
		}
		return fw.Close()
	}

	// Read in batches
	batchSize := 10000
	written := 0
	for written < numRows {
		n := batchSize
		if written+n > numRows {
			n = numRows - written
		}
		rows := make([]oldGameFeatures, n)
		if err := pr.Read(&rows); err != nil {
			closePW()
			return written, fmt.Errorf("read batch at row %d: %w", written, err)
		}
		for i := range rows {
			old := &rows[i]
			nMoves := len(old.Piece)
			newPlayer := make([]int32, nMoves)
			compMove := make([]int32, nMoves)
			for j := 0; j < nMoves; j++ {
				newPlayer[j] = remapPlayer(old.Player[j])
				compMove[j] = compositeMoveID(
					old.Piece[j], old.From[j], old.To[j], old.Capture[j],
					old.Promotion[j], old.Check[j], old.Castle[j], newPlayer[j],
				)
			}
			out := newGameFeatures{
				Piece:         old.Piece,
				From:          old.From,
				To:            old.To,
				Capture:       old.Capture,
				Promotion:     old.Promotion,
				Check:         old.Check,
				Castle:        old.Castle,
				Player:        newPlayer,
				UCIMove:       old.UCIMove,
				CompositeMove: compMove,
			}
			if err := pw.Write(out); err != nil {
				closePW()
				return written, fmt.Errorf("write row %d: %w", written+i, err)
			}
		}
		written += n
	}

	if err := closePW(); err != nil {
		return written, fmt.Errorf("close writer: %w", err)
	}
	return written, nil
}

func main() {
	var inDir, outDir, compositeVocabPath string
	flag.StringVar(&inDir, "in", "", "input directory with old-format shard-*.parquet files")
	flag.StringVar(&outDir, "out", "", "output directory for new-format shards")
	flag.StringVar(&compositeVocabPath, "composite-vocab", "composite_vocab.txt", "path to composite vocab")
	flag.Parse()

	if inDir == "" || outDir == "" {
		fmt.Fprintln(os.Stderr, "usage: convertcache -in <old-cache-dir> -out <new-cache-dir>")
		os.Exit(2)
	}

	if err := loadCompositeVocab(compositeVocabPath); err != nil {
		fmt.Fprintf(os.Stderr, "error: could not load composite vocab from %s: %v\n", compositeVocabPath, err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "loaded composite vocab: %d tuples\n", len(compositeVocab))

	if err := os.MkdirAll(outDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "mkdir: %v\n", err)
		os.Exit(1)
	}

	matches, err := filepath.Glob(filepath.Join(inDir, "shard-*.parquet"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "glob: %v\n", err)
		os.Exit(1)
	}
	sort.Strings(matches)

	if len(matches) == 0 {
		fmt.Fprintln(os.Stderr, "no shard-*.parquet files found in input directory")
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "converting %d shards: %s -> %s\n", len(matches), inDir, outDir)
	start := time.Now()
	totalRows := 0

	for i, inPath := range matches {
		base := filepath.Base(inPath)
		outPath := filepath.Join(outDir, base)
		n, err := convertShard(inPath, outPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error converting %s: %v\n", base, err)
			os.Exit(1)
		}
		totalRows += n
		fmt.Fprintf(os.Stderr, "  [%d/%d] %s: %d games\n", i+1, len(matches), base, n)
	}

	elapsed := time.Since(start).Truncate(time.Millisecond)
	fmt.Fprintf(os.Stderr, "done. %d shards, %d games, %s\n", len(matches), totalRows, elapsed)
}
