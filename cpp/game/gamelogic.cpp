#include "../game/gamelogic.h"

/*
 * gamelogic.cpp
 * Logics of game rules
 * Some other game logics are in board.h/cpp
 *
 * Gomoku as a representive
 */

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

bool Board::checkConnection(int8_t* buf, Player pla, bool includeJumpConnection) const {
  (void)includeJumpConnection;
  std::fill(buf, buf + Board::MAX_ARR_SIZE, 0);
  std::vector<Loc> stack;
  stack.reserve(Board::MAX_PLAY_SIZE);

  for(int y = 0; y < y_size; y++) {
    for(int x = 0; x < x_size; x++) {
      Loc start = Location::getLoc(x, y, x_size);
      if(colors[start] != pla || buf[start] != 0)
        continue;

      int sidesTouched = 0;
      buf[start] = 1;
      stack.clear();
      stack.push_back(start);

      while(!stack.empty()) {
        Loc loc = stack.back();
        stack.pop_back();

        sidesTouched |= sideMask(loc);
        if(sidesTouched == 7)
          return true;

        Loc adjacent[6];
        int numAdjacent = getAdjacentLocs(loc, adjacent);
        for(int adjIdx = 0; adjIdx < numAdjacent; adjIdx++) {
          Loc adj = adjacent[adjIdx];
          if(isOnBoard(adj) && colors[adj] == pla && buf[adj] == 0) {
            buf[adj] = 1;
            stack.push_back(adj);
          }
        }
      }
    }
  }

  return false;
}

void Board::initCaptureTable() {
  //first, dead locations
  //3 basic shapes
  {
    //  x x
    // x . x
    //  # #
    int t[6] = {1, 1, 1, 1, 0, 0};
    // two # loc
    for(int a = 0; a < 3; a++)
      for(int b = 0; b < 3; b++) {
        t[4] = a;
        t[5] = b;
        int id = t[0] + t[1] * 4 + t[2] * 16 + t[3] * 64 + t[4] * 256 + t[5] * 1024;
        CAPTURE_TABLE[id] = 1;
      }
  }
  {
    //  x x
    // x . #
    //  # o
    int t[6] = {1, 1, 1, 0, 2, 0};
    // two # loc
    for(int a = 0; a < 3; a++)
      for(int b = 0; b < 3; b++) {
        t[3] = a;
        t[5] = b;
        int id = t[0] + t[1] * 4 + t[2] * 16 + t[3] * 64 + t[4] * 256 + t[5] * 1024;
        CAPTURE_TABLE[id] = 1;
      }
  }
  {
    //  x #
    // x . o
    //  # o
    int t[6] = {1, 1, 0, 2, 2, 0};
    // two # loc
    for(int a = 0; a < 3; a++)
      for(int b = 0; b < 3; b++) {
        t[2] = a;
        t[5] = b;
        int id = t[0] + t[1] * 4 + t[2] * 16 + t[3] * 64 + t[4] * 256 + t[5] * 1024;
        CAPTURE_TABLE[id] = 1;
      }
  }

  // 6 rotates
  for(int id = 0; id < 4096; id++) {
    if(CAPTURE_TABLE[id] != 1)
      continue;
    int t = id;
    for(int rot = 0; rot < 6; rot++) {
      t = 1024 * (t % 4) + t / 4;
      CAPTURE_TABLE[t] = 1;
    }
  }
  // color inverse
  for(int id = 0; id < 4096; id++) {
    if(CAPTURE_TABLE[id] != 1)
      continue;
    int a = id;

    int t[6];
    for (int i = 0; i < 6; i++)
    {
      int c = a % 4;
      t[i] = c == 1 ? 2 : c == 2 ? 1 : 0;
      a /= 4;
    }
    int id2 = t[0] + t[1] * 4 + t[2] * 16 + t[3] * 64 + t[4] * 256 + t[5] * 1024;
    CAPTURE_TABLE[id2] = 1;
  }

  //next, captured or dominated 
  //replace any stone of dead shapes with C_EMPTY
  
  for(int id = 0; id < 4096; id++) {
    if(CAPTURE_TABLE[id] != 1)
      continue;
    for (int i = 0; i < 6; i++)
    {
      int id2 = id & (~(3 << (2 * i)));//replace each stone with 0
      if(CAPTURE_TABLE[id2] == 0)
        CAPTURE_TABLE[id2] = 2;
    }
  }

  //finally, consider "any" color
  //replace each location with 3

  for(int id = 0; id < 4096; id++) {
    if(CAPTURE_TABLE[id] == 0)
      continue;

    for(int k = 0; k < 64; k++) {
      int m = k;
      int c = 0;
      for (int i = 0; i < 6; i++)
      {
        c *= 4;
        if (m % 2)
        {
          c |= 3;
        }
        m /= 2;
      }
      int id2 = id | c;//replace each loc with 3

      if(CAPTURE_TABLE[id2] == 0)
        CAPTURE_TABLE[id2] = CAPTURE_TABLE[id];
      else if(CAPTURE_TABLE[id2] == 2 && CAPTURE_TABLE[id] == 1)
        CAPTURE_TABLE[id2] = 1;
    }

  }
  IS_CAPTURETABLE_INITALIZED = true;
}

bool Board::isDeadOrCaptured(Loc loc) const {
  assert(IS_CAPTURETABLE_INITALIZED);

  if(!isOnBoard(loc))
    return true;
  if(colors[loc] != C_EMPTY)
    return true;
  if(shape == BoardShape::BentY)
    return false;

  int x0 = Location::getX(loc, x_size);
  int y0 = Location::getY(loc, x_size);

  const int dxs[6] = {0, 1, 1, 0, -1, -1};
  const int dys[6] = {-1, -1, 0, 1, 1, 0};

  int surroundings = 0;
  for(int d = 0; d < 6; d++) {
    int x = x0 + dxs[d];
    int y = y0 + dys[d];
    if(!isPlayablePoint(x, y))
      return false;
    Loc adj = Location::getLoc(x, y, x_size);
    Color c = colors[adj];
    surroundings *= 4;
    surroundings += c;
  }

  return CAPTURE_TABLE[surroundings] != 0;
}

Color GameLogic::checkWinnerAfterPlayed(
  const Board& board,
  const BoardHistory& hist,
  Player pla,
  Loc loc,
  int8_t* bufferForCheckingWinner) {
  (void)hist;
  if(board.checkConnection(bufferForCheckingWinner, pla, false))
    return pla;

  if(loc == Board::PASS_LOC)
    return getOpp(pla);  //pass is not allowed

  //check maxmoves
  if (hist.rules.maxMoves > 0)
  {
    int currentMovenum = board.numStonesOnBoard();
    if(currentMovenum >= hist.rules.maxMoves)
      return C_EMPTY;
  }
  

  return C_WALL;
}

GameLogic::ResultsBeforeNN::ResultsBeforeNN() {
  inited = false;
  winner = C_WALL;
  myOnlyLoc = Board::NULL_LOC;
}

void GameLogic::ResultsBeforeNN::init(const Board& board, const BoardHistory& hist, Color nextPlayer) {
  //not used in Y
  (void)board;
  (void)hist;
  (void)nextPlayer;
  if(inited)
    return;
  inited = true;

  return;
}
