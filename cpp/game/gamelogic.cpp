#include "../game/gamelogic.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

bool Board::checkConnection(int8_t* buf, Player pla) const {
  std::fill(buf,buf+x_size*y_size,0);
  vector<Loc> stack;
  for(int y = 0; y < y_size; y += 2) {
    for(int x = 0; x < x_size; x++) {
      bool source = pla == P_BLACK ? y == 0 : x == 0;
      Loc loc = Location::getLoc(x,y,x_size);
      if(source && colors[loc] == pla) {
        buf[x+y*x_size] = 1;
        stack.push_back(loc);
      }
    }
  }

  while(!stack.empty()) {
    Loc loc = stack.back();
    stack.pop_back();
    int x = Location::getX(loc,x_size);
    int y = Location::getY(loc,x_size);
    if((pla == P_BLACK && y == y_size-1) || (pla == P_WHITE && x == x_size-1))
      return true;

    const int orthDx[4] = {-1,1,0,0};
    const int orthDy[4] = {0,0,-2,2};
    for(int i = 0; i < 4; i++) {
      int nx = x + orthDx[i];
      int ny = y + orthDy[i];
      if(nx < 0 || nx >= x_size || ny < 0 || ny >= y_size)
        continue;
      Loc next = Location::getLoc(nx,ny,x_size);
      if(colors[next] == pla && buf[nx+ny*x_size] == 0) {
        buf[nx+ny*x_size] = 1;
        stack.push_back(next);
      }
    }

    const int diagDx[4] = {-1,1,-1,1};
    const int diagDy[4] = {-2,-2,2,2};
    for(int i = 0; i < 4; i++) {
      int nx = x + diagDx[i];
      int ny = y + diagDy[i];
      if(nx < 0 || nx >= x_size || ny < 0 || ny >= y_size)
        continue;
      int diamondX = std::min(x,nx);
      int diamondY = (y+ny)/2;
      Loc diamond = Location::getLoc(diamondX,diamondY,x_size);
      Loc next = Location::getLoc(nx,ny,x_size);
      CrosscutDirection direction = crosscutDirections[diamond];
      bool directionMatches = direction == CROSSCUT_BOTH ||
        (direction == CROSSCUT_BACKSLASH && (nx-x)*(ny-y) > 0) ||
        (direction == CROSSCUT_SLASH && (nx-x)*(ny-y) < 0);
      if(colors[diamond] == pla && directionMatches && colors[next] == pla && buf[nx+ny*x_size] == 0) {
        buf[nx+ny*x_size] = 1;
        stack.push_back(next);
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
  if(loc == PASS_LOC)
    return true;
  Loc physicalLoc = getPhysicalLoc(loc);
  return !isOnBoard(physicalLoc) || colors[physicalLoc] != C_EMPTY;

}

Color GameLogic::checkWinnerAfterPlayed(
  const Board& board,
  const BoardHistory& hist,
  Player pla,
  Loc loc,
  int8_t* bufferForCheckingWinner) {
  if(board.checkConnection(bufferForCheckingWinner, pla))
    return pla;

  if(loc == Board::PASS_LOC)
    return getOpp(pla);  //pass is not allowed

  if(QuaxVariantIO::hasDirectionalCrosscuts(board.variant) && board.numStonesOnBoard() >= board.playableArea())
    return C_EMPTY;

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
  (void)board;
  (void)hist;
  (void)nextPlayer;
  if(inited)
    return;
  inited = true;

  return;
}
