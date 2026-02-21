// compositevocab enumerates every valid 8-feature tuple for chess move
// prediction and prints them one per line, tab-separated.  Line number
// (0-indexed) is the composite vocab ID.
//
// Features (matching san_features.py FEATURE_ORDER after step removal):
//
//	piece, from_square, to_square, capture, promotion, check, castle, player
//
// Feature IDs are 0-indexed (0 = NULL for each feature).
//
// "Valid" means geometrically reachable for the given piece type.
// We enumerate all legal move geometries per piece, crossed with:
//   - capture: {0=NULL, 1=x}
//   - check:   {0=NULL, 1=+, 2=#}
//   - player:  {1=white, 2=black}
//
// Promotion is only nonzero for pawns reaching the back rank.
// Castle tuples have fixed piece=K, fixed from/to, capture=0, promotion=0.
package main

import (
	"fmt"
	"os"
	"sort"
	"strings"
)

func inBounds(f, r int) bool { return f >= 0 && f < 8 && r >= 0 && r < 8 }

// squareID: NULL=0, a1=1, b1=2, ..., h8=64 (rank-major, matching Go preprocessor)
func squareID(file, rank int) int { return 1 + rank*8 + file }

// pieceID: NULL=0, P=1, N=2, B=3, R=4, Q=5, K=6
const (
	pPawn   = 1
	pKnight = 2
	pBishop = 3
	pRook   = 4
	pQueen  = 5
	pKing   = 6
)

// promotionID: NULL=0, N=1, B=2, R=3, Q=4
const (
	promoNull   = 0
	promoKnight = 1
	promoBishop = 2
	promoRook   = 3
	promoQueen  = 4
)

// castleID: NULL=0, O-O=1, O-O-O=2
const (
	castleNull = 0
	castleKS   = 1
	castleQS   = 2
)

// captureID: NULL=0, x=1
const (
	capNull = 0
	capX    = 1
)

// checkIDs: NULL=0, +=1, #=2
// playerIDs: white=1, black=2

type tuple struct {
	piece      int
	fromSquare int
	toSquare   int
	capture    int
	promotion  int
	check      int
	castle     int
	player     int
}

func (t tuple) String() string {
	return fmt.Sprintf("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d",
		t.piece, t.fromSquare, t.toSquare, t.capture, t.promotion, t.check, t.castle, t.player)
}

func (t tuple) sortKey() string {
	return fmt.Sprintf("%02d-%02d-%02d-%d-%d-%d-%d-%d",
		t.piece, t.fromSquare, t.toSquare, t.capture, t.promotion, t.check, t.castle, t.player)
}

func main() {
	// Collect base move geometries: (piece, from, to, promoOptions)
	type baseMoveGeom struct {
		piece  int
		fromSq int
		toSq   int
		promos []int // list of valid promotion IDs (just [0] for non-promoting)
	}

	var geoms []baseMoveGeom
	seen := map[[3]int]bool{} // deduplicate (piece, from, to)

	addGeom := func(piece, fromF, fromR, toF, toR int, promos []int) {
		fid := squareID(fromF, fromR)
		tid := squareID(toF, toR)
		key := [3]int{piece, fid, tid}
		if seen[key] {
			return
		}
		seen[key] = true
		geoms = append(geoms, baseMoveGeom{piece: piece, fromSq: fid, toSq: tid, promos: promos})
	}

	noPromo := []int{promoNull}
	allPromos := []int{promoKnight, promoBishop, promoRook, promoQueen}

	for f := 0; f < 8; f++ {
		for r := 0; r < 8; r++ {
			// --- Knight ---
			for _, d := range [][2]int{
				{1, 2}, {2, 1}, {2, -1}, {1, -2},
				{-1, -2}, {-2, -1}, {-2, 1}, {-1, 2},
			} {
				tf, tr := f+d[0], r+d[1]
				if inBounds(tf, tr) {
					addGeom(pKnight, f, r, tf, tr, noPromo)
				}
			}

			// --- King (1-square moves, non-castling) ---
			for _, d := range [][2]int{
				{1, 1}, {1, 0}, {1, -1}, {0, 1},
				{0, -1}, {-1, 1}, {-1, 0}, {-1, -1},
			} {
				tf, tr := f+d[0], r+d[1]
				if inBounds(tf, tr) {
					addGeom(pKing, f, r, tf, tr, noPromo)
				}
			}

			// --- Rook (straights) ---
			for _, d := range [][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}} {
				tf, tr := f+d[0], r+d[1]
				for inBounds(tf, tr) {
					addGeom(pRook, f, r, tf, tr, noPromo)
					tf += d[0]
					tr += d[1]
				}
			}

			// --- Bishop (diagonals) ---
			for _, d := range [][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}} {
				tf, tr := f+d[0], r+d[1]
				for inBounds(tf, tr) {
					addGeom(pBishop, f, r, tf, tr, noPromo)
					tf += d[0]
					tr += d[1]
				}
			}

			// --- Queen (straights + diagonals) ---
			for _, d := range [][2]int{
				{1, 0}, {-1, 0}, {0, 1}, {0, -1},
				{1, 1}, {1, -1}, {-1, 1}, {-1, -1},
			} {
				tf, tr := f+d[0], r+d[1]
				for inBounds(tf, tr) {
					addGeom(pQueen, f, r, tf, tr, noPromo)
					tf += d[0]
					tr += d[1]
				}
			}

			// --- White pawns ---
			if r >= 1 && r <= 6 {
				// Non-promoting push
				if r <= 5 {
					addGeom(pPawn, f, r, f, r+1, noPromo)
				}
				// Double push from rank 2 (index 1)
				if r == 1 {
					addGeom(pPawn, f, r, f, r+2, noPromo)
				}
				// Non-promoting captures
				if r <= 5 {
					if f > 0 {
						addGeom(pPawn, f, r, f-1, r+1, noPromo)
					}
					if f < 7 {
						addGeom(pPawn, f, r, f+1, r+1, noPromo)
					}
				}
				// Promotion from rank 7 (index 6)
				if r == 6 {
					addGeom(pPawn, f, r, f, r+1, allPromos)
					if f > 0 {
						addGeom(pPawn, f, r, f-1, r+1, allPromos)
					}
					if f < 7 {
						addGeom(pPawn, f, r, f+1, r+1, allPromos)
					}
				}
			}

			// --- Black pawns ---
			if r >= 1 && r <= 6 {
				// Non-promoting push
				if r >= 2 {
					addGeom(pPawn, f, r, f, r-1, noPromo)
				}
				// Double push from rank 7 (index 6)
				if r == 6 {
					addGeom(pPawn, f, r, f, r-2, noPromo)
				}
				// Non-promoting captures
				if r >= 2 {
					if f > 0 {
						addGeom(pPawn, f, r, f-1, r-1, noPromo)
					}
					if f < 7 {
						addGeom(pPawn, f, r, f+1, r-1, noPromo)
					}
				}
				// Promotion from rank 2 (index 1)
				if r == 1 {
					addGeom(pPawn, f, r, f, r-1, allPromos)
					if f > 0 {
						addGeom(pPawn, f, r, f-1, r-1, allPromos)
					}
					if f < 7 {
						addGeom(pPawn, f, r, f+1, r-1, allPromos)
					}
				}
			}
		}
	}

	// Expand base geometries into full 8-feature tuples.
	// For each geometry, cross with: capture {0,1}, check {0,1,2}, player {1,2,3,4}.
	// Castle tuples handled separately below.
	tupleSet := map[string]tuple{}

	addTuple := func(t tuple) {
		key := t.sortKey()
		tupleSet[key] = t
	}

	for _, g := range geoms {
		for _, promo := range g.promos {
			for _, cap := range []int{capNull, capX} {
				// Pawns: double-push can't be a capture
				if g.piece == pPawn && cap == capX {
					fromRank := (g.fromSq - 1) / 8
					toRank := (g.toSq - 1) / 8
					fromFile := (g.fromSq - 1) % 8
					toFile := (g.toSq - 1) % 8
					// Straight push (same file) can't capture
					if fromFile == toFile {
						continue
					}
					// Diagonal non-capture can't happen for pawns
					_ = fromRank
					_ = toRank
				}
				// Pawns: non-capture must be same file
				if g.piece == pPawn && cap == capNull {
					fromFile := (g.fromSq - 1) % 8
					toFile := (g.toSq - 1) % 8
					if fromFile != toFile {
						continue
					}
				}
				for _, chk := range []int{0, 1, 2} {
					for _, pl := range []int{1, 2} {
						addTuple(tuple{
							piece:      g.piece,
							fromSquare: g.fromSq,
							toSquare:   g.toSq,
							capture:    cap,
							promotion:  promo,
							check:      chk,
							castle:     castleNull,
							player:     pl,
						})
					}
				}
			}
		}
	}

	// Castling tuples: piece=K, fixed from/to, capture=0, promotion=0
	castleMoves := []struct {
		fromSq int
		toSq   int
		kind   int
	}{
		{squareID(4, 0), squareID(6, 0), castleKS}, // white O-O: e1->g1
		{squareID(4, 0), squareID(2, 0), castleQS}, // white O-O-O: e1->c1
		{squareID(4, 7), squareID(6, 7), castleKS}, // black O-O: e8->g8
		{squareID(4, 7), squareID(2, 7), castleQS}, // black O-O-O: e8->c8
	}
	for _, cm := range castleMoves {
		for _, chk := range []int{0, 1, 2} {
			for _, pl := range []int{1, 2} {
				addTuple(tuple{
					piece:      pKing,
					fromSquare: cm.fromSq,
					toSquare:   cm.toSq,
					capture:    capNull,
					promotion:  promoNull,
					check:      chk,
					castle:     cm.kind,
					player:     pl,
				})
			}
		}
	}

	// Sort and output
	keys := make([]string, 0, len(tupleSet))
	for k := range tupleSet {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var lines []string
	for _, k := range keys {
		lines = append(lines, tupleSet[k].String())
	}

	fmt.Print(strings.Join(lines, "\n"))
	if len(lines) > 0 {
		fmt.Println()
	}
	fmt.Fprintf(os.Stderr, "composite vocab size: %d\n", len(lines))
}
