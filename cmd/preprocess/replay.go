package main

import (
	"math/bits"
	"strings"

	"github.com/notnil/chess"
)

const (
	liteWhite int8 = 0
	liteBlack int8 = 1
)

const (
	litePawn   int8 = 1
	liteKnight int8 = 2
	liteBishop int8 = 3
	liteRook   int8 = 4
	liteQueen  int8 = 5
	liteKing   int8 = 6
)

const (
	castleWhiteKing  uint8 = 1 << 0
	castleWhiteQueen uint8 = 1 << 1
	castleBlackKing  uint8 = 1 << 2
	castleBlackQueen uint8 = 1 << 3
)

var backRank = [8]int8{liteRook, liteKnight, liteBishop, liteQueen, liteKing, liteBishop, liteKnight, liteRook}

type liteSANMove struct {
	piece   int8
	to      int
	capture bool
	promo   int8
	disFile int8
	disRank int8
	castle  int8
	check   int32
}

type liteBoard struct {
	sq     [64]int8
	bb     [2][6]uint64
	occ    [2]uint64
	all    uint64
	kingSq [2]int8
	side   int8
	castle uint8
	enPass int8
}

func liteAbs(v int8) int8 {
	if v < 0 {
		return -v
	}
	return v
}

func liteColorOfCode(code int8) int8 {
	if code > 0 {
		return liteWhite
	}
	return liteBlack
}

func liteCode(color int8, piece int8) int8 {
	if color == liteBlack {
		return -piece
	}
	return piece
}

func liteBit(sq int) uint64 {
	return uint64(1) << uint(sq)
}

func liteFile(sq int) int {
	return sq & 7
}

func liteRank(sq int) int {
	return sq >> 3
}

func liteInBounds(file int, rank int) bool {
	return file >= 0 && file < 8 && rank >= 0 && rank < 8
}

func liteSquareIDFromIndex(sq int) int32 {
	return int32(sq + 1)
}

func litePromotionID(piece int8) int32 {
	switch piece {
	case liteKnight:
		return 1
	case liteBishop:
		return 2
	case liteRook:
		return 3
	case liteQueen:
		return 4
	default:
		return nullID
	}
}

func litePlayerID(turn int8) int32 {
	if turn == liteWhite {
		return playerWhite
	}
	return playerBlack
}

func (b *liteBoard) reset() {
	b.sq = [64]int8{}
	b.bb = [2][6]uint64{}
	b.occ = [2]uint64{}
	b.all = 0
	b.kingSq[0] = -1
	b.kingSq[1] = -1
	b.side = liteWhite
	b.castle = castleWhiteKing | castleWhiteQueen | castleBlackKing | castleBlackQueen
	b.enPass = -1

	for file := 0; file < 8; file++ {
		b.addPiece(file, liteWhite, backRank[file])
		b.addPiece(8+file, liteWhite, litePawn)
		b.addPiece(48+file, liteBlack, litePawn)
		b.addPiece(56+file, liteBlack, backRank[file])
	}
}

func (b *liteBoard) addPiece(sq int, color int8, piece int8) {
	code := liteCode(color, piece)
	mask := liteBit(sq)
	b.sq[sq] = code
	b.bb[color][piece-1] |= mask
	b.occ[color] |= mask
	b.all |= mask
	if piece == liteKing {
		b.kingSq[color] = int8(sq)
	}
}

func (b *liteBoard) removePiece(sq int) int8 {
	code := b.sq[sq]
	if code == 0 {
		return 0
	}
	color := liteColorOfCode(code)
	piece := liteAbs(code)
	mask := liteBit(sq)
	b.sq[sq] = 0
	b.bb[color][piece-1] &^= mask
	b.occ[color] &^= mask
	b.all &^= mask
	if piece == liteKing {
		b.kingSq[color] = -1
	}
	return code
}

func (b *liteBoard) clearCastleForMover(from int, piece int8, color int8) {
	if piece == liteKing {
		if color == liteWhite {
			b.castle &^= castleWhiteKing | castleWhiteQueen
		} else {
			b.castle &^= castleBlackKing | castleBlackQueen
		}
		return
	}
	if piece != liteRook {
		return
	}
	switch from {
	case 0:
		b.castle &^= castleWhiteQueen
	case 7:
		b.castle &^= castleWhiteKing
	case 56:
		b.castle &^= castleBlackQueen
	case 63:
		b.castle &^= castleBlackKing
	}
}

func (b *liteBoard) clearCastleForCapturedRook(to int, captured int8) {
	if liteAbs(captured) != liteRook {
		return
	}
	switch to {
	case 0:
		b.castle &^= castleWhiteQueen
	case 7:
		b.castle &^= castleWhiteKing
	case 56:
		b.castle &^= castleBlackQueen
	case 63:
		b.castle &^= castleBlackKing
	}
}

func (b *liteBoard) pathClear(from int, to int, df int, dr int) bool {
	f := liteFile(from) + df
	r := liteRank(from) + dr
	tf := liteFile(to)
	tr := liteRank(to)
	for f != tf || r != tr {
		if !liteInBounds(f, r) {
			return false
		}
		idx := r*8 + f
		if b.sq[idx] != 0 {
			return false
		}
		f += df
		r += dr
	}
	return true
}

func (b *liteBoard) canReach(from int, to int, piece int8, color int8, capture bool) bool {
	if from == to {
		return false
	}
	ff := liteFile(from)
	fr := liteRank(from)
	tf := liteFile(to)
	tr := liteRank(to)
	df := tf - ff
	dr := tr - fr

	switch piece {
	case litePawn:
		if color == liteWhite {
			if capture {
				if dr != 1 || (df != -1 && df != 1) {
					return false
				}
				if b.sq[to] != 0 {
					return liteColorOfCode(b.sq[to]) != color
				}
				return b.enPass == int8(to)
			}
			if df != 0 {
				return false
			}
			if dr == 1 {
				return b.sq[to] == 0
			}
			if dr == 2 && fr == 1 {
				mid := from + 8
				return b.sq[mid] == 0 && b.sq[to] == 0
			}
			return false
		}
		if capture {
			if dr != -1 || (df != -1 && df != 1) {
				return false
			}
			if b.sq[to] != 0 {
				return liteColorOfCode(b.sq[to]) != color
			}
			return b.enPass == int8(to)
		}
		if df != 0 {
			return false
		}
		if dr == -1 {
			return b.sq[to] == 0
		}
		if dr == -2 && fr == 6 {
			mid := from - 8
			return b.sq[mid] == 0 && b.sq[to] == 0
		}
		return false
	case liteKnight:
		adx := df
		if adx < 0 {
			adx = -adx
		}
		ady := dr
		if ady < 0 {
			ady = -ady
		}
		return (adx == 1 && ady == 2) || (adx == 2 && ady == 1)
	case liteBishop:
		adx := df
		if adx < 0 {
			adx = -adx
		}
		ady := dr
		if ady < 0 {
			ady = -ady
		}
		if adx != ady || adx == 0 {
			return false
		}
		stepF := 1
		if df < 0 {
			stepF = -1
		}
		stepR := 1
		if dr < 0 {
			stepR = -1
		}
		return b.pathClear(from, to, stepF, stepR)
	case liteRook:
		if df != 0 && dr != 0 {
			return false
		}
		stepF := 0
		if df > 0 {
			stepF = 1
		} else if df < 0 {
			stepF = -1
		}
		stepR := 0
		if dr > 0 {
			stepR = 1
		} else if dr < 0 {
			stepR = -1
		}
		return b.pathClear(from, to, stepF, stepR)
	case liteQueen:
		if df == 0 || dr == 0 {
			stepF := 0
			if df > 0 {
				stepF = 1
			} else if df < 0 {
				stepF = -1
			}
			stepR := 0
			if dr > 0 {
				stepR = 1
			} else if dr < 0 {
				stepR = -1
			}
			return b.pathClear(from, to, stepF, stepR)
		}
		adx := df
		if adx < 0 {
			adx = -adx
		}
		ady := dr
		if ady < 0 {
			ady = -ady
		}
		if adx != ady {
			return false
		}
		stepF := 1
		if df < 0 {
			stepF = -1
		}
		stepR := 1
		if dr < 0 {
			stepR = -1
		}
		return b.pathClear(from, to, stepF, stepR)
	case liteKing:
		adx := df
		if adx < 0 {
			adx = -adx
		}
		ady := dr
		if ady < 0 {
			ady = -ady
		}
		return adx <= 1 && ady <= 1
	default:
		return false
	}
}

func (b *liteBoard) isSquareAttackedBy(sq int, attacker int8) bool {
	file := liteFile(sq)
	rank := liteRank(sq)

	// Pawns.
	if attacker == liteWhite {
		if file > 0 && rank > 0 {
			code := b.sq[sq-9]
			if code == liteCode(attacker, litePawn) {
				return true
			}
		}
		if file < 7 && rank > 0 {
			code := b.sq[sq-7]
			if code == liteCode(attacker, litePawn) {
				return true
			}
		}
	} else {
		if file > 0 && rank < 7 {
			code := b.sq[sq+7]
			if code == liteCode(attacker, litePawn) {
				return true
			}
		}
		if file < 7 && rank < 7 {
			code := b.sq[sq+9]
			if code == liteCode(attacker, litePawn) {
				return true
			}
		}
	}

	// Knights.
	knightOffsets := [8][2]int{{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}}
	for _, off := range knightOffsets {
		f := file + off[0]
		r := rank + off[1]
		if !liteInBounds(f, r) {
			continue
		}
		idx := r*8 + f
		if b.sq[idx] == liteCode(attacker, liteKnight) {
			return true
		}
	}

	// King.
	kingOffsets := [8][2]int{{1, 1}, {1, 0}, {1, -1}, {0, 1}, {0, -1}, {-1, 1}, {-1, 0}, {-1, -1}}
	for _, off := range kingOffsets {
		f := file + off[0]
		r := rank + off[1]
		if !liteInBounds(f, r) {
			continue
		}
		idx := r*8 + f
		if b.sq[idx] == liteCode(attacker, liteKing) {
			return true
		}
	}

	// Diagonals (bishop/queen).
	diag := [4][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}
	for _, d := range diag {
		f := file + d[0]
		r := rank + d[1]
		for liteInBounds(f, r) {
			idx := r*8 + f
			code := b.sq[idx]
			if code != 0 {
				if liteColorOfCode(code) == attacker {
					pt := liteAbs(code)
					if pt == liteBishop || pt == liteQueen {
						return true
					}
				}
				break
			}
			f += d[0]
			r += d[1]
		}
	}

	// Straights (rook/queen).
	straight := [4][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	for _, d := range straight {
		f := file + d[0]
		r := rank + d[1]
		for liteInBounds(f, r) {
			idx := r*8 + f
			code := b.sq[idx]
			if code != 0 {
				if liteColorOfCode(code) == attacker {
					pt := liteAbs(code)
					if pt == liteRook || pt == liteQueen {
						return true
					}
				}
				break
			}
			f += d[0]
			r += d[1]
		}
	}

	return false
}

func (b *liteBoard) inCheck(color int8) bool {
	kingSq := b.kingSq[color]
	if kingSq < 0 {
		return false
	}
	return b.isSquareAttackedBy(int(kingSq), color^1)
}

func (b *liteBoard) applyNormal(from int, to int, piece int8, promo int8, capture bool) bool {
	color := b.side
	code := b.sq[from]
	if code == 0 || liteColorOfCode(code) != color || liteAbs(code) != piece {
		return false
	}

	target := b.sq[to]
	enPassCapture := false
	if piece == litePawn && capture && target == 0 && b.enPass == int8(to) {
		enPassCapture = true
	}

	if capture {
		if enPassCapture {
			capSq := to - 8
			if color == liteBlack {
				capSq = to + 8
			}
			capCode := b.sq[capSq]
			if capCode == 0 || liteColorOfCode(capCode) == color || liteAbs(capCode) != litePawn {
				return false
			}
			b.removePiece(capSq)
		} else {
			if target == 0 || liteColorOfCode(target) == color {
				return false
			}
			b.clearCastleForCapturedRook(to, target)
			b.removePiece(to)
		}
	} else if target != 0 {
		return false
	}

	b.clearCastleForMover(from, piece, color)
	b.removePiece(from)
	finalPiece := piece
	if piece == litePawn && promo != 0 {
		finalPiece = promo
	}
	b.addPiece(to, color, finalPiece)

	b.enPass = -1
	if piece == litePawn && promo == 0 {
		if color == liteWhite && to-from == 16 {
			b.enPass = int8(from + 8)
		} else if color == liteBlack && from-to == 16 {
			b.enPass = int8(from - 8)
		}
	}
	b.side = color ^ 1
	return true
}

func (b *liteBoard) applyCastle(kind int8) (int, int, bool) {
	color := b.side
	if b.inCheck(color) {
		return 0, 0, false
	}
	if color == liteWhite {
		if b.sq[4] != liteCode(liteWhite, liteKing) {
			return 0, 0, false
		}
		if kind == 1 {
			if (b.castle&castleWhiteKing) == 0 || b.sq[7] != liteCode(liteWhite, liteRook) {
				return 0, 0, false
			}
			if b.sq[5] != 0 || b.sq[6] != 0 {
				return 0, 0, false
			}
			if b.isSquareAttackedBy(5, liteBlack) || b.isSquareAttackedBy(6, liteBlack) {
				return 0, 0, false
			}
			b.removePiece(4)
			b.removePiece(7)
			b.addPiece(6, liteWhite, liteKing)
			b.addPiece(5, liteWhite, liteRook)
			b.castle &^= castleWhiteKing | castleWhiteQueen
			b.enPass = -1
			b.side = liteBlack
			return 4, 6, true
		}
		if (b.castle&castleWhiteQueen) == 0 || b.sq[0] != liteCode(liteWhite, liteRook) {
			return 0, 0, false
		}
		if b.sq[1] != 0 || b.sq[2] != 0 || b.sq[3] != 0 {
			return 0, 0, false
		}
		if b.isSquareAttackedBy(3, liteBlack) || b.isSquareAttackedBy(2, liteBlack) {
			return 0, 0, false
		}
		b.removePiece(4)
		b.removePiece(0)
		b.addPiece(2, liteWhite, liteKing)
		b.addPiece(3, liteWhite, liteRook)
		b.castle &^= castleWhiteKing | castleWhiteQueen
		b.enPass = -1
		b.side = liteBlack
		return 4, 2, true
	}

	if b.sq[60] != liteCode(liteBlack, liteKing) {
		return 0, 0, false
	}
	if kind == 1 {
		if (b.castle&castleBlackKing) == 0 || b.sq[63] != liteCode(liteBlack, liteRook) {
			return 0, 0, false
		}
		if b.sq[61] != 0 || b.sq[62] != 0 {
			return 0, 0, false
		}
		if b.isSquareAttackedBy(61, liteWhite) || b.isSquareAttackedBy(62, liteWhite) {
			return 0, 0, false
		}
		b.removePiece(60)
		b.removePiece(63)
		b.addPiece(62, liteBlack, liteKing)
		b.addPiece(61, liteBlack, liteRook)
		b.castle &^= castleBlackKing | castleBlackQueen
		b.enPass = -1
		b.side = liteWhite
		return 60, 62, true
	}
	if (b.castle&castleBlackQueen) == 0 || b.sq[56] != liteCode(liteBlack, liteRook) {
		return 0, 0, false
	}
	if b.sq[57] != 0 || b.sq[58] != 0 || b.sq[59] != 0 {
		return 0, 0, false
	}
	if b.isSquareAttackedBy(59, liteWhite) || b.isSquareAttackedBy(58, liteWhite) {
		return 0, 0, false
	}
	b.removePiece(60)
	b.removePiece(56)
	b.addPiece(58, liteBlack, liteKing)
	b.addPiece(59, liteBlack, liteRook)
	b.castle &^= castleBlackKing | castleBlackQueen
	b.enPass = -1
	b.side = liteWhite
	return 60, 58, true
}

func (b *liteBoard) resolveAndApply(m liteSANMove) (int, int, int8, bool, int8, bool) {
	if m.castle != 0 {
		from, to, ok := b.applyCastle(m.castle)
		if !ok {
			return 0, 0, 0, false, 0, false
		}
		return from, to, liteKing, false, 0, true
	}

	color := b.side
	target := b.sq[m.to]
	if target != 0 && liteColorOfCode(target) == color {
		return 0, 0, 0, false, 0, false
	}

	var chosenFrom = -1
	var chosenPiece int8
	var chosenCapture bool

	candidateLegal := func(from int, piece int8, capture bool) bool {
		if m.disFile >= 0 && int(m.disFile) != liteFile(from) {
			return false
		}
		if m.disRank >= 0 && int(m.disRank) != liteRank(from) {
			return false
		}
		if !b.canReach(from, m.to, piece, color, capture) {
			return false
		}

		tmp := *b
		if !tmp.applyNormal(from, m.to, piece, m.promo, capture) {
			return false
		}
		if tmp.inCheck(color) {
			return false
		}
		return true
	}

	if m.piece == 0 {
		if m.disFile < 0 || m.disRank < 0 {
			return 0, 0, 0, false, 0, false
		}
		from := int(m.disRank)*8 + int(m.disFile)
		code := b.sq[from]
		if code == 0 || liteColorOfCode(code) != color {
			return 0, 0, 0, false, 0, false
		}
		piece := liteAbs(code)
		capture := m.capture
		if !capture {
			capture = target != 0 || (piece == litePawn && b.enPass == int8(m.to) && liteFile(from) != liteFile(m.to))
		}
		if !candidateLegal(from, piece, capture) {
			return 0, 0, 0, false, 0, false
		}
		chosenFrom = from
		chosenPiece = piece
		chosenCapture = capture
	} else {
		bb := b.bb[color][m.piece-1]
		for bb != 0 {
			from := bits.TrailingZeros64(bb)
			bb &= bb - 1
			if !candidateLegal(from, m.piece, m.capture) {
				continue
			}
			if chosenFrom != -1 {
				// Duplicate legal source means ambiguous SAN for this board.
				return 0, 0, 0, false, 0, false
			}
			chosenFrom = from
			chosenPiece = m.piece
			chosenCapture = m.capture
		}
	}

	if chosenFrom == -1 {
		return 0, 0, 0, false, 0, false
	}
	if !b.applyNormal(chosenFrom, m.to, chosenPiece, m.promo, chosenCapture) {
		return 0, 0, 0, false, 0, false
	}
	return chosenFrom, m.to, chosenPiece, chosenCapture, m.promo, true
}

func liteParseSquare(text string) (int, bool) {
	if len(text) != 2 {
		return 0, false
	}
	file := int(text[0] - 'a')
	rank := int(text[1] - '1')
	if !liteInBounds(file, rank) {
		return 0, false
	}
	return rank*8 + file, true
}

func litePromoPiece(ch byte) (int8, bool) {
	switch ch {
	case 'N':
		return liteKnight, true
	case 'B':
		return liteBishop, true
	case 'R':
		return liteRook, true
	case 'Q':
		return liteQueen, true
	default:
		return 0, false
	}
}

func liteIsUCI(token string) bool {
	if len(token) != 4 && len(token) != 5 {
		return false
	}
	_, ok1 := liteParseSquare(token[:2])
	_, ok2 := liteParseSquare(token[2:4])
	if !ok1 || !ok2 {
		return false
	}
	if len(token) == 5 {
		p := token[4]
		return p == 'q' || p == 'r' || p == 'b' || p == 'n' || p == 'Q' || p == 'R' || p == 'B' || p == 'N'
	}
	return true
}

func parseLiteSAN(token string) (liteSANMove, bool) {
	s := liteSANMove{
		piece:   litePawn,
		to:      -1,
		disFile: -1,
		disRank: -1,
	}
	if token == "" || token == "..." || token == "e.p." || token == "ep" {
		return s, false
	}

	// UCI fallback.
	if liteIsUCI(token) {
		from, _ := liteParseSquare(token[:2])
		to, _ := liteParseSquare(token[2:4])
		s.piece = 0
		s.to = to
		s.disFile = int8(liteFile(from))
		s.disRank = int8(liteRank(from))
		if len(token) == 5 {
			promo, ok := litePromoPiece(byte(strings.ToUpper(token[4:5])[0]))
			if !ok {
				return s, false
			}
			s.promo = promo
		}
		return s, true
	}

	for len(token) > 0 {
		last := token[len(token)-1]
		if last == '#' {
			s.check = 2
			token = token[:len(token)-1]
			continue
		}
		if last == '+' {
			if s.check == 0 {
				s.check = 1
			}
			token = token[:len(token)-1]
			continue
		}
		break
	}

	if token == "O-O" || token == "0-0" {
		s.piece = liteKing
		s.castle = 1
		return s, true
	}
	if token == "O-O-O" || token == "0-0-0" {
		s.piece = liteKing
		s.castle = 2
		return s, true
	}

	if eq := strings.IndexByte(token, '='); eq >= 0 {
		if eq+1 >= len(token) {
			return s, false
		}
		promo, ok := litePromoPiece(token[eq+1])
		if !ok {
			return s, false
		}
		s.promo = promo
		token = token[:eq]
	}

	if len(token) < 2 {
		return s, false
	}
	to, ok := liteParseSquare(token[len(token)-2:])
	if !ok {
		return s, false
	}
	s.to = to
	prefix := token[:len(token)-2]

	if strings.IndexByte(prefix, 'x') >= 0 {
		s.capture = true
		prefix = strings.ReplaceAll(prefix, "x", "")
	}

	if len(prefix) > 0 {
		switch prefix[0] {
		case 'K':
			s.piece = liteKing
			prefix = prefix[1:]
		case 'Q':
			s.piece = liteQueen
			prefix = prefix[1:]
		case 'R':
			s.piece = liteRook
			prefix = prefix[1:]
		case 'B':
			s.piece = liteBishop
			prefix = prefix[1:]
		case 'N':
			s.piece = liteKnight
			prefix = prefix[1:]
		}
	}

	if len(prefix) == 1 {
		c := prefix[0]
		if c >= 'a' && c <= 'h' {
			s.disFile = int8(c - 'a')
		} else if c >= '1' && c <= '8' {
			s.disRank = int8(c - '1')
		} else {
			return s, false
		}
	} else if len(prefix) >= 2 {
		f := prefix[0]
		r := prefix[1]
		if f >= 'a' && f <= 'h' && r >= '1' && r <= '8' {
			s.disFile = int8(f - 'a')
			s.disRank = int8(r - '1')
		} else {
			return s, false
		}
	}

	return s, true
}

func buildFeaturesFromMovesLite(movetext string, winner chess.Color) (*gameFeatures, bool) {
	var board liteBoard
	board.reset()
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
		if isMoveNumberToken(token) || token == "..." {
			continue
		}

		moveStr := stripMoveSuffix(token)
		if moveStr == "" || moveStr == "..." {
			continue
		}
		parsed, ok := parseLiteSAN(moveStr)
		if !ok {
			return nil, false
		}

		turn := board.side
		from, to, piece, capture, promo, ok := board.resolveAndApply(parsed)
		if !ok {
			return nil, false
		}

		pID := int32(piece)
		fID := liteSquareIDFromIndex(from)
		tID := liteSquareIDFromIndex(to)
		var capID int32
		if capture {
			capID = 1
		}
		promoID := litePromotionID(promo)
		chkID := parsed.check
		castID := int32(parsed.castle)
		plID := litePlayerID(turn)

		features.Piece = append(features.Piece, pID)
		features.From = append(features.From, fID)
		features.To = append(features.To, tID)
		features.Capture = append(features.Capture, capID)
		features.Promotion = append(features.Promotion, promoID)
		features.Check = append(features.Check, chkID)
		features.Castle = append(features.Castle, castID)
		features.Player = append(features.Player, plID)
		features.UCIMove = append(features.UCIMove, uciMoveID(from, to, litePromoChar(promo)))
		features.CompositeMove = append(features.CompositeMove, compositeMoveID(
			pID, fID, tID, capID, promoID, chkID, castID, plID,
		))
	}

	if len(features.Piece) == 0 {
		return nil, false
	}
	return features, true
}
