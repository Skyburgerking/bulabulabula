# ============================================
# GEDA 象棋AI - 完整稳定版 v3.2.2
# 功能：记忆池优化 + 延迟信用完整 + 策略效用可视化 + 复盘学习 + 重置无效局
# 调优：探索模式记忆调用概率 50%，记忆池阈值 0.20
# 新增：输棋后自动复盘最后三步，将好走法的基因片段标记为高价值
#       按 R 重置对局视为无效局，不触发延迟信用学习
# ============================================

import pygame
import numpy as np
import random
import sys
import time
import os
import pickle
from collections import defaultdict, deque, Counter
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any
import math
import hashlib
import platform

# 初始化pygame
pygame.init()

# ============================================
# 1. 全局常量与配置
# ============================================

BOARD_SIZE = 10
GRID_SIZE = 60
BOARD_WIDTH = 9 * GRID_SIZE
BOARD_HEIGHT = 10 * GRID_SIZE
MARGIN = 20
SCREEN_WIDTH = BOARD_WIDTH + 400
SCREEN_HEIGHT = BOARD_HEIGHT + 100
INFO_PANEL_WIDTH = 380
INFO_PANEL_X = BOARD_WIDTH + 10

# 颜色定义
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
DARK_RED = (180, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 180, 0)
BLUE = (0, 120, 255)
LIGHT_BLUE = (100, 200, 255)
YELLOW = (255, 255, 0)
DARK_YELLOW = (200, 200, 0)
PURPLE = (180, 0, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
ORANGE = (255, 165, 0)
GRAY = (100, 100, 100)
LIGHT_GRAY = (180, 180, 180)
DARK_GRAY = (50, 50, 50)
BOARD_BROWN = (205, 133, 63)
BOARD_LIGHT = (255, 239, 213)
BOARD_DARK = (160, 82, 45)

RED_PIECE_COLOR = (255, 50, 50)
BLACK_PIECE_COLOR = (50, 50, 50)
HIGHLIGHT_COLOR = (255, 255, 100, 180)

# GEDA基因系统常量
GENE_BASES = ['A', 'G', 'C', 'T', 'X']
GENE_COMPLEMENTS = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'X': 'X'}

# 长期记忆文件路径
MEMORY_DIR = "geda_memory"
RED_MEMORY_FILE = os.path.join(MEMORY_DIR, "red_ai_memory_v3.pkl")
BLACK_MEMORY_FILE = os.path.join(MEMORY_DIR, "black_ai_memory_v3.pkl")
GAME_LOG_DIR = os.path.join(MEMORY_DIR, "game_logs")

# 游戏状态
class GameState(Enum):
    PLAYING = 1
    RED_WIN = 2
    BLACK_WIN = 3
    DRAW = 4

# 玩家类型
class PlayerType(Enum):
    HUMAN = 1
    GEDA_AI = 2

# 棋子类型枚举
class PieceType(Enum):
    GENERAL = 1
    ADVISOR = 2
    ELEPHANT = 3
    HORSE = 4
    CHARIOT = 5
    CANNON = 6
    SOLDIER = 7


# ============================================
# 2. 棋子类（完整实现）
# ============================================
class ChessPiece:
    def __init__(self, piece_type: PieceType, is_red: bool, position: Tuple[int, int]):
        self.piece_type = piece_type
        self.is_red = is_red
        self.position = position
        self.captured = False
        self.pressure_score = 0.0
        self.threat_level = 0.0
        self.mobility_score = 0.0

        self.value_map = {
            PieceType.GENERAL: 10000,
            PieceType.CHARIOT: 900,
            PieceType.CANNON: 450,
            PieceType.HORSE: 400,
            PieceType.ELEPHANT: 200,
            PieceType.ADVISOR: 200,
            PieceType.SOLDIER: 100
        }
        self.value = self.value_map[piece_type]
        self.position_weights = self._init_position_weights()

    def _init_position_weights(self):
        weights = np.ones((10, 9))
        if self.piece_type == PieceType.GENERAL:
            if self.is_red:
                for y in range(7, 10):
                    for x in range(3, 6):
                        weights[y, x] = 2.0
                weights[9, 4] = 3.0
            else:
                for y in range(0, 3):
                    for x in range(3, 6):
                        weights[y, x] = 2.0
                weights[0, 4] = 3.0
        elif self.piece_type == PieceType.CHARIOT:
            for y in range(10):
                for x in range(9):
                    center_dist = abs(x - 4) + abs(y - 4.5)
                    weights[y, x] += 1.8 - center_dist / 5.0
                    if self.is_red and y <= 2:
                        weights[y, x] += 0.3
                    elif not self.is_red and y >= 7:
                        weights[y, x] += 0.3
        elif self.piece_type == PieceType.HORSE:
            for y in range(10):
                for x in range(9):
                    center_dist = abs(x - 4) + abs(y - 4.5)
                    weights[y, x] += 1.0 - center_dist / 6.0
                    if (x <= 1 or x >= 7) and (y <= 1 or y >= 8):
                        weights[y, x] *= 0.7
        elif self.piece_type == PieceType.CANNON:
            for y in range(10):
                for x in range(9):
                    if self.is_red and y >= 6:
                        weights[y, x] += 0.8
                    elif not self.is_red and y <= 3:
                        weights[y, x] += 0.8
                    if (self.is_red and y == 6 and 3 <= x <= 5) or \
                       (not self.is_red and y == 3 and 3 <= x <= 5):
                        weights[y, x] += 1.0
                    if (self.is_red and y == 5) or (not self.is_red and y == 4):
                        weights[y, x] += 0.5
        elif self.piece_type == PieceType.SOLDIER:
            for y in range(10):
                for x in range(9):
                    if self.is_red:
                        if y <= 4:
                            weights[y, x] += 2.0
                            dist_to_king = abs(x - 4) + abs(y - 0)
                            weights[y, x] += 1.5 - dist_to_king / 12.0
                        else:
                            weights[y, x] = 0.8
                    else:
                        if y >= 5:
                            weights[y, x] += 2.0
                            dist_to_king = abs(x - 4) + abs(y - 9)
                            weights[y, x] += 1.5 - dist_to_king / 12.0
                        else:
                            weights[y, x] = 0.8
        elif self.piece_type in (PieceType.ELEPHANT, PieceType.ADVISOR):
            if self.is_red:
                for y in range(5, 10):
                    for x in range(9):
                        weights[y, x] *= 1.1
                for y in range(0, 5):
                    for x in range(9):
                        weights[y, x] *= 0.3
            else:
                for y in range(0, 5):
                    for x in range(9):
                        weights[y, x] *= 1.1
                for y in range(5, 10):
                    for x in range(9):
                        weights[y, x] *= 0.3
        return weights

    def update_pressure_analysis(self, board: 'ChessBoard'):
        x, y = self.position
        if self.captured:
            self.pressure_score = 0.0
            self.threat_level = 0.0
            self.mobility_score = 0.0
            return
        threat_value = 0
        for piece in board.pieces:
            if not piece.captured and piece.is_red != self.is_red:
                if (x, y) in piece.get_moves(board):
                    threat_value += piece.value
        max_possible_threat = 1500
        self.threat_level = min(1.0, threat_value / max_possible_threat)
        possible_moves = self.get_moves(board)
        self.mobility_score = min(1.0, len(possible_moves) / 25.0)
        value_factor = self.value / 10000.0
        if self.piece_type == PieceType.GENERAL:
            self.pressure_score = self.threat_level * 0.9 + (1 - self.mobility_score) * 0.05 + value_factor * 0.05
        else:
            self.pressure_score = self.threat_level * 0.7 + (1 - self.mobility_score) * 0.15 + value_factor * 0.15

    def get_position_weight(self, x: int, y: int) -> float:
        if 0 <= y < 10 and 0 <= x < 9:
            return self.position_weights[y, x]
        return 1.0

    def get_symbol(self):
        symbols_red = {
            PieceType.GENERAL: "帥",
            PieceType.ADVISOR: "仕",
            PieceType.ELEPHANT: "相",
            PieceType.HORSE: "馬",
            PieceType.CHARIOT: "車",
            PieceType.CANNON: "炮",
            PieceType.SOLDIER: "兵"
        }
        symbols_black = {
            PieceType.GENERAL: "將",
            PieceType.ADVISOR: "士",
            PieceType.ELEPHANT: "象",
            PieceType.HORSE: "馬",
            PieceType.CHARIOT: "車",
            PieceType.CANNON: "砲",
            PieceType.SOLDIER: "卒"
        }
        return symbols_red[self.piece_type] if self.is_red else symbols_black[self.piece_type]

    def get_moves(self, board: 'ChessBoard') -> List[Tuple[int, int]]:
        x, y = self.position
        moves = []
        if self.captured:
            return moves
        if self.piece_type == PieceType.GENERAL:
            if self.is_red:
                palace = [(3,7),(4,7),(5,7),(3,8),(4,8),(5,8),(3,9),(4,9),(5,9)]
            else:
                palace = [(3,0),(4,0),(5,0),(3,1),(4,1),(5,1),(3,2),(4,2),(5,2)]
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = x+dx, y+dy
                if (nx, ny) in palace:
                    tp = board.get_piece_at(nx, ny)
                    if tp is None or tp.is_red != self.is_red:
                        moves.append((nx, ny))
        elif self.piece_type == PieceType.ADVISOR:
            if self.is_red:
                palace = [(3,7),(4,7),(5,7),(3,8),(4,8),(5,8),(3,9),(4,9),(5,9)]
            else:
                palace = [(3,0),(4,0),(5,0),(3,1),(4,1),(5,1),(3,2),(4,2),(5,2)]
            for dx, dy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                nx, ny = x+dx, y+dy
                if (nx, ny) in palace:
                    tp = board.get_piece_at(nx, ny)
                    if tp is None or tp.is_red != self.is_red:
                        moves.append((nx, ny))
        elif self.piece_type == PieceType.ELEPHANT:
            directions = [(2,2),(2,-2),(-2,2),(-2,-2)]
            blocks = [(1,1),(1,-1),(-1,1),(-1,-1)]
            for i, (dx, dy) in enumerate(directions):
                nx, ny = x+dx, y+dy
                if 0 <= nx < 9 and 0 <= ny < 10:
                    if (self.is_red and ny <= 4) or (not self.is_red and ny >= 5):
                        continue
                    bx, by = x+blocks[i][0], y+blocks[i][1]
                    if 0 <= bx < 9 and 0 <= by < 10 and board.get_piece_at(bx, by) is None:
                        tp = board.get_piece_at(nx, ny)
                        if tp is None or tp.is_red != self.is_red:
                            moves.append((nx, ny))
        elif self.piece_type == PieceType.HORSE:
            horse_moves = [(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)]
            horse_legs = [(0,1),(1,0),(0,1),(-1,0),(0,-1),(1,0),(0,-1),(-1,0)]
            for i, (dx, dy) in enumerate(horse_moves):
                nx, ny = x+dx, y+dy
                if 0 <= nx < 9 and 0 <= ny < 10:
                    lx, ly = x+horse_legs[i][0], y+horse_legs[i][1]
                    if 0 <= lx < 9 and 0 <= ly < 10 and board.get_piece_at(lx, ly) is None:
                        tp = board.get_piece_at(nx, ny)
                        if tp is None or tp.is_red != self.is_red:
                            moves.append((nx, ny))
        elif self.piece_type == PieceType.CHARIOT:
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = x+dx, y+dy
                while 0 <= nx < 9 and 0 <= ny < 10:
                    tp = board.get_piece_at(nx, ny)
                    if tp is None:
                        moves.append((nx, ny))
                        nx += dx
                        ny += dy
                    else:
                        if tp.is_red != self.is_red:
                            moves.append((nx, ny))
                        break
        elif self.piece_type == PieceType.CANNON:
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = x+dx, y+dy
                found = False
                while 0 <= nx < 9 and 0 <= ny < 10:
                    tp = board.get_piece_at(nx, ny)
                    if not found:
                        if tp is None:
                            moves.append((nx, ny))
                        else:
                            found = True
                    else:
                        if tp is not None:
                            if tp.is_red != self.is_red:
                                moves.append((nx, ny))
                            break
                    nx += dx
                    ny += dy
        elif self.piece_type == PieceType.SOLDIER:
            if self.is_red:
                if y > 4:
                    directions = [(0,-1)]
                else:
                    directions = [(0,-1),(1,0),(-1,0)]
            else:
                if y < 5:
                    directions = [(0,1)]
                else:
                    directions = [(0,1),(1,0),(-1,0)]
            for dx, dy in directions:
                nx, ny = x+dx, y+dy
                if 0 <= nx < 9 and 0 <= ny < 10:
                    tp = board.get_piece_at(nx, ny)
                    if tp is None or tp.is_red != self.is_red:
                        moves.append((nx, ny))
        return moves

    def __repr__(self):
        color = "红" if self.is_red else "黑"
        type_names = {
            PieceType.GENERAL: "将帅",
            PieceType.ADVISOR: "士",
            PieceType.ELEPHANT: "象",
            PieceType.HORSE: "马",
            PieceType.CHARIOT: "车",
            PieceType.CANNON: "炮",
            PieceType.SOLDIER: "兵卒"
        }
        return f"{color}{type_names[self.piece_type]}({self.position})"


# ============================================
# 3. 棋盘类（完整实现）
# ============================================
class ChessBoard:
    def __init__(self):
        self.board = [[None for _ in range(9)] for _ in range(10)]
        self.pieces = []
        self.current_player = True  # True:红方
        self.game_state = GameState.PLAYING
        self.move_history = []
        self.pressure_map = np.zeros((10, 9), dtype=np.float32)
        self.setup_board()

    def setup_board(self):
        """初始化棋盘，放置所有棋子"""
        self.board = [[None for _ in range(9)] for _ in range(10)]
        self.pieces = []

        # 红方棋子
        red_positions = [
            (PieceType.CHARIOT, (0,9)), (PieceType.HORSE, (1,9)), (PieceType.ELEPHANT, (2,9)),
            (PieceType.ADVISOR, (3,9)), (PieceType.GENERAL, (4,9)), (PieceType.ADVISOR, (5,9)),
            (PieceType.ELEPHANT, (6,9)), (PieceType.HORSE, (7,9)), (PieceType.CHARIOT, (8,9)),
            (PieceType.CANNON, (1,7)), (PieceType.CANNON, (7,7)),
            (PieceType.SOLDIER, (0,6)), (PieceType.SOLDIER, (2,6)), (PieceType.SOLDIER, (4,6)),
            (PieceType.SOLDIER, (6,6)), (PieceType.SOLDIER, (8,6))
        ]
        # 黑方棋子
        black_positions = [
            (PieceType.CHARIOT, (0,0)), (PieceType.HORSE, (1,0)), (PieceType.ELEPHANT, (2,0)),
            (PieceType.ADVISOR, (3,0)), (PieceType.GENERAL, (4,0)), (PieceType.ADVISOR, (5,0)),
            (PieceType.ELEPHANT, (6,0)), (PieceType.HORSE, (7,0)), (PieceType.CHARIOT, (8,0)),
            (PieceType.CANNON, (1,2)), (PieceType.CANNON, (7,2)),
            (PieceType.SOLDIER, (0,3)), (PieceType.SOLDIER, (2,3)), (PieceType.SOLDIER, (4,3)),
            (PieceType.SOLDIER, (6,3)), (PieceType.SOLDIER, (8,3))
        ]

        for pt, pos in red_positions:
            piece = ChessPiece(pt, True, pos)
            self.pieces.append(piece)
            self.board[pos[1]][pos[0]] = piece
        for pt, pos in black_positions:
            piece = ChessPiece(pt, False, pos)
            self.pieces.append(piece)
            self.board[pos[1]][pos[0]] = piece

        self.update_pressure_map()

    def update_pressure_map(self):
        self.pressure_map = np.zeros((10, 9), dtype=np.float32)
        for piece in self.pieces:
            if not piece.captured:
                piece.update_pressure_analysis(self)
                x, y = piece.position
                self.pressure_map[y, x] = piece.pressure_score
        temp = self.pressure_map.copy()
        for y in range(10):
            for x in range(9):
                if self.pressure_map[y, x] > 0:
                    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < 9 and 0 <= ny < 10:
                            temp[ny, nx] += self.pressure_map[y, x] * 0.2
        self.pressure_map = np.clip(temp, 0, 1)

    def get_piece_at(self, x: int, y: int) -> Optional[ChessPiece]:
        if 0 <= x < 9 and 0 <= y < 10:
            return self.board[y][x]
        return None

    def get_general(self, is_red: bool) -> Optional[ChessPiece]:
        for p in self.pieces:
            if not p.captured and p.is_red == is_red and p.piece_type == PieceType.GENERAL:
                return p
        return None

    def is_in_check(self, is_red: bool) -> bool:
        general = self.get_general(is_red)
        if not general:
            return False
        gx, gy = general.position
        for piece in self.pieces:
            if not piece.captured and piece.is_red != is_red:
                if (gx, gy) in piece.get_moves(self):
                    return True
        return False

    def is_legal_move(self, piece: ChessPiece, target: Tuple[int, int]) -> bool:
        if target not in piece.get_moves(self):
            return False
        old_x, old_y = piece.position
        target_piece = self.get_piece_at(target[0], target[1])
        if target_piece:
            target_piece.captured = True
            target_piece.position = (-1, -1)
            self.board[target[1]][target[0]] = None
        self.board[old_y][old_x] = None
        self.board[target[1]][target[0]] = piece
        piece.position = (target[0], target[1])
        in_check = self.is_in_check(piece.is_red)
        piece.position = (old_x, old_y)
        self.board[old_y][old_x] = piece
        self.board[target[1]][target[0]] = target_piece
        if target_piece:
            target_piece.captured = False
            target_piece.position = (target[0], target[1])
        return not in_check

    def get_legal_moves(self, player_red: bool) -> List[Tuple[ChessPiece, Tuple[int, int]]]:
        moves = []
        for piece in self.pieces:
            if not piece.captured and piece.is_red == player_red:
                for move in piece.get_moves(self):
                    if self.is_legal_move(piece, move):
                        moves.append((piece, move))
        return moves

    def move_piece(self, piece: ChessPiece, new_pos: Tuple[int, int]) -> bool:
        if not self.is_legal_move(piece, new_pos):
            return False
        old_x, old_y = piece.position
        new_x, new_y = new_pos
        captured = self.get_piece_at(new_x, new_y)
        move_record = {
            'piece': piece,
            'from': (old_x, old_y),
            'to': (new_x, new_y),
            'captured': captured
        }
        if captured:
            captured.captured = True
            captured.position = (-1, -1)
            self.board[new_y][new_x] = None
        self.board[old_y][old_x] = None
        self.board[new_y][new_x] = piece
        piece.position = (new_x, new_y)
        self.move_history.append(move_record)
        self.current_player = not self.current_player
        self.update_pressure_map()
        self.check_game_state()
        return True

    def undo_move(self) -> bool:
        if not self.move_history:
            return False
        last = self.move_history.pop()
        piece = last['piece']
        from_pos = last['from']
        to_pos = last['to']
        captured = last['captured']
        self.board[to_pos[1]][to_pos[0]] = None
        self.board[from_pos[1]][from_pos[0]] = piece
        piece.position = from_pos
        if captured:
            captured.captured = False
            captured.position = to_pos
            self.board[to_pos[1]][to_pos[0]] = captured
        self.current_player = not self.current_player
        self.game_state = GameState.PLAYING
        self.update_pressure_map()
        return True

    def check_game_state(self):
        red_g = self.get_general(True)
        black_g = self.get_general(False)
        if red_g is None:
            self.game_state = GameState.BLACK_WIN
        elif black_g is None:
            self.game_state = GameState.RED_WIN
        else:
            self.game_state = GameState.PLAYING

    def evaluate_board_detailed(self, for_red: bool) -> float:
        score = 0.0
        for piece in self.pieces:
            if not piece.captured:
                val = piece.value * piece.position_weights[piece.position[1], piece.position[0]]
                if piece.piece_type == PieceType.SOLDIER:
                    if (piece.is_red and piece.position[1] <= 4) or (not piece.is_red and piece.position[1] >= 5):
                        val *= 2.0
                score += val if piece.is_red else -val
        red_mobility = len(self.get_legal_moves(True))
        black_mobility = len(self.get_legal_moves(False))
        score += (red_mobility - black_mobility) * 8
        red_threat = self._calc_threat_value(True)
        black_threat = self._calc_threat_value(False)
        score += (red_threat - black_threat) * 0.3
        red_protected = self._calc_protected_value(True)
        black_protected = self._calc_protected_value(False)
        score += (red_protected - black_protected) * 0.2
        red_g = self.get_general(True)
        black_g = self.get_general(False)
        if red_g:
            score -= red_g.pressure_score * 600
        if black_g:
            score += black_g.pressure_score * 600
        score += self._calc_coordination(True) - self._calc_coordination(False)
        score += self._calc_control(True) - self._calc_control(False)
        red_river = sum(1 for p in self.pieces if not p.captured and p.is_red and p.piece_type == PieceType.SOLDIER and p.position[1] <= 4)
        black_river = sum(1 for p in self.pieces if not p.captured and not p.is_red and p.piece_type == PieceType.SOLDIER and p.position[1] >= 5)
        score += (red_river - black_river) * 30
        red_elephant = sum(1 for p in self.pieces if not p.captured and p.is_red and p.piece_type == PieceType.ELEPHANT)
        red_advisor = sum(1 for p in self.pieces if not p.captured and p.is_red and p.piece_type == PieceType.ADVISOR)
        black_elephant = sum(1 for p in self.pieces if not p.captured and not p.is_red and p.piece_type == PieceType.ELEPHANT)
        black_advisor = sum(1 for p in self.pieces if not p.captured and not p.is_red and p.piece_type == PieceType.ADVISOR)
        score += (red_elephant + red_advisor - black_elephant - black_advisor) * 20
        return score if for_red else -score

    def _calc_threat_value(self, is_red: bool) -> float:
        threat = 0.0
        enemy_general = self.get_general(not is_red)
        if not enemy_general:
            return 0
        for piece in self.pieces:
            if not piece.captured and piece.is_red == is_red:
                moves = piece.get_moves(self)
                if enemy_general.position in moves:
                    threat += piece.value * 2.0
                else:
                    for opp in self.pieces:
                        if not opp.captured and opp.is_red != is_red and opp.value >= 400:
                            if opp.position in moves:
                                threat += opp.value * 0.3
        return threat

    def _calc_protected_value(self, is_red: bool) -> float:
        protected = 0.0
        for piece in self.pieces:
            if not piece.captured and piece.is_red == is_red:
                if self.is_protected(piece):
                    protected += piece.value * 0.3
        return protected

    def is_protected(self, piece: ChessPiece) -> bool:
        x, y = piece.position
        for p in self.pieces:
            if not p.captured and p.is_red == piece.is_red and p != piece:
                if (x, y) in p.get_moves(self):
                    return True
        return False

    def _calc_coordination(self, is_red: bool) -> float:
        score = 0.0
        chariots = [p for p in self.pieces if not p.captured and p.is_red == is_red and p.piece_type == PieceType.CHARIOT]
        if len(chariots) >= 2:
            for i in range(len(chariots)):
                for j in range(i+1, len(chariots)):
                    if chariots[i].position[0] == chariots[j].position[0] or chariots[i].position[1] == chariots[j].position[1]:
                        score += 50
        horses = [p for p in self.pieces if not p.captured and p.is_red == is_red and p.piece_type == PieceType.HORSE]
        cannons = [p for p in self.pieces if not p.captured and p.is_red == is_red and p.piece_type == PieceType.CANNON]
        for h in horses:
            for c in cannons:
                if abs(h.position[0]-c.position[0]) + abs(h.position[1]-c.position[1]) <= 2:
                    score += 20
        return score

    def _calc_control(self, is_red: bool) -> float:
        control = 0.0
        center = [(4,4),(4,5),(5,4),(5,5)]
        for x, y in center:
            piece = self.get_piece_at(x, y)
            if piece and piece.is_red == is_red:
                control += 10
            else:
                for p in self.pieces:
                    if not p.captured and p.is_red == is_red and (x, y) in p.get_moves(self):
                        control += 3
        for y in [4,5]:
            for x in range(9):
                if self.get_piece_at(x, y) is None:
                    for p in self.pieces:
                        if not p.captured and p.is_red == is_red and (x, y) in p.get_moves(self):
                            control += 1
        return control

    def extract_situation_signature(self) -> str:
        sig_parts = []
        for y in range(10):
            for x in range(9):
                piece = self.board[y][x]
                if piece:
                    sig_parts.append(f"{piece.piece_type.value}{'R' if piece.is_red else 'B'}{x}{y}")
        sig_parts.sort()
        sig_parts.append(f"turn{'R' if self.current_player else 'B'}")
        full = "".join(sig_parts)
        return hashlib.md5(full.encode()).hexdigest()[:16]

    def get_pressure_analysis(self) -> Dict:
        analysis = {
            'red_pressure': 0.0,
            'black_pressure': 0.0,
            'piece_analysis': [],
            'high_pressure_pieces': [],
            'general_status': {
                'red': {'pressure': 0.0, 'threat': 0.0, 'safe': True},
                'black': {'pressure': 0.0, 'threat': 0.0, 'safe': True}
            }
        }
        for piece in self.pieces:
            if piece.captured:
                continue
            piece_analysis = {
                'piece': str(piece),
                'pressure': piece.pressure_score,
                'threat': piece.threat_level,
                'mobility': piece.mobility_score
            }
            analysis['piece_analysis'].append(piece_analysis)
            if piece.pressure_score > 0.7:
                analysis['high_pressure_pieces'].append(str(piece))
            if piece.is_red:
                analysis['red_pressure'] += piece.pressure_score
                if piece.piece_type == PieceType.GENERAL:
                    analysis['general_status']['red'] = {
                        'pressure': piece.pressure_score,
                        'threat': piece.threat_level,
                        'safe': piece.pressure_score < 0.5 and piece.threat_level < 0.3
                    }
            else:
                analysis['black_pressure'] += piece.pressure_score
                if piece.piece_type == PieceType.GENERAL:
                    analysis['general_status']['black'] = {
                        'pressure': piece.pressure_score,
                        'threat': piece.threat_level,
                        'safe': piece.pressure_score < 0.5 and piece.threat_level < 0.3
                    }
        piece_count = len([p for p in self.pieces if not p.captured])
        if piece_count > 0:
            analysis['red_pressure'] /= piece_count
            analysis['black_pressure'] /= piece_count
        return analysis

    def copy(self) -> 'ChessBoard':
        new = ChessBoard()
        new.board = [[None for _ in range(9)] for _ in range(10)]
        new.pieces = []
        new.current_player = self.current_player
        new.game_state = self.game_state
        new.move_history = self.move_history.copy()
        new.pressure_map = self.pressure_map.copy()
        for piece in self.pieces:
            new_piece = ChessPiece(piece.piece_type, piece.is_red, piece.position)
            new_piece.captured = piece.captured
            new_piece.pressure_score = piece.pressure_score
            new_piece.threat_level = piece.threat_level
            new_piece.mobility_score = piece.mobility_score
            new.pieces.append(new_piece)
            if not piece.captured:
                new.board[piece.position[1]][piece.position[0]] = new_piece
        return new


# ============================================
# 4. GEDA基因系统 - v3.2.2 复盘学习版
# ============================================
class GEDA_GeneSystem:
    def __init__(self, is_red: bool, memory_file: str = None):
        self.is_red = is_red
        self.bases = GENE_BASES
        self.complement_map = GENE_COMPLEMENTS

        self.gene_chain = self._initialize_gene_chain(40)
        self.complementary_chain = [self.complement_map[b] for b in self.gene_chain]

        self.doubt_gene_chain = self._initialize_doubt_gene_chain()

        self.doubt_strategy_utility = defaultdict(lambda: 0.5)
        self.last_doubt_strategy = None

        self.memory_gene_pool = []
        self.memory_capacity = 30

        self.success_streak = 0
        self.failure_streak = 0
        self.performance_history = deque(maxlen=20)
        self.mode_switch_threshold = 0.7

        self.strategy_mode = "balanced"
        self.exploration_rate = 0.15
        self.confidence = 0.7

        self.long_term_memory = {
            'game_patterns': {},
            'opponent_style': {}
        }
        self.failure_memory = defaultdict(list)
        self.success_memory = defaultdict(list)

        self.gene_segment_utility = defaultdict(float)

        self.doubt_states = deque(maxlen=30)
        self.doubt_effectiveness = 0.5

        self.rule_patterns = defaultdict(lambda: {'success':0, 'failure':0, 'total':0, 'confidence':0.0})
        self.pressure_correlations = Counter()

        self.last_decision_info = {
            'initial_move': None,
            'final_move': None,
            'doubt_triggered': False,
            'doubt_score': 0.0,
            'doubt_reason': '',
            'delta': 0.0,
            'memory_hit': False,
            'memory_success_rate': 0.0,
            'strategy_mode': 'balanced',
            'environment_pressure': 0.5,
            'general_pressure': 0.0,
            'general_safe': True,
            'move_description': '',
            'timestamp': 0
        }

        self.game_moves = []

        self.memory_file = memory_file if memory_file else (RED_MEMORY_FILE if is_red else BLACK_MEMORY_FILE)

        self.success_count = 0
        self.total_attempts = 0
        self.recent_success_rate = 0.0

        self.load_long_term_memory()
        print(f"GEDA基因系统初始化 - {'红方' if is_red else '黑方'}，基因长度: {len(self.gene_chain)}")

    def _initialize_gene_chain(self, length):
        segments = [
            (10, ['A','A','G','T','C','A','G','X','T','A']),
            (10, ['G','C','T','G','C','G','T','G','C','T']),
            (10, ['C','C','T','A','C','C','T','A','C','C']),
            (10, ['T','A','G','C','T','A','G','C','T','A'])
        ]
        chain = []
        for seg_len, pat in segments:
            for i in range(min(seg_len, length - len(chain))):
                chain.append(pat[i % len(pat)])
        while len(chain) < length:
            chain.append(random.choice(self.bases))
        return chain[:length]

    def _initialize_doubt_gene_chain(self):
        weights = {'A':0.1, 'G':0.2, 'C':0.2, 'T':0.3, 'X':0.2}
        chain = [random.choices(list(weights.keys()), weights=list(weights.values()))[0] for _ in range(20)]
        return {'primary': chain, 'complementary': [self.complement_map[b] for b in chain]}

    def _update_strategy_mode(self):
        if len(self.performance_history) < 5:
            return
        recent = list(self.performance_history)[-10:] if len(self.performance_history) >= 10 else list(self.performance_history)
        recent_success = sum(recent) / len(recent) if recent else 0
        if recent_success < 0.2 or self.failure_streak >= 3:
            self.strategy_mode = "explorative"
            self.exploration_rate = min(0.4, self.exploration_rate * 1.2 + 0.05)
            print(f" [宏观正反馈] 连续失败{self.failure_streak}次，进入探索模式，探索率={self.exploration_rate:.2f}")
        elif recent_success > 0.6 or self.success_streak >= 3:
            self.strategy_mode = "exploitative"
            self.exploration_rate = max(0.05, self.exploration_rate * 0.8)
            print(f" [宏观正反馈] 连续成功{self.success_streak}次，进入开发模式，探索率={self.exploration_rate:.2f}")
        else:
            self.strategy_mode = "balanced"
            self.exploration_rate = max(0.1, self.exploration_rate * 0.98)
        if self.total_attempts > 0:
            self.recent_success_rate = self.success_count / self.total_attempts

    def _add_to_memory_pool(self, segment, utility):
        for i, (seg, util, cnt, ts) in enumerate(self.memory_gene_pool):
            if seg == segment:
                new_util = (util * cnt + utility) / (cnt + 1)
                self.memory_gene_pool[i] = (seg, new_util, cnt + 1, self.total_attempts)
                return
        if len(self.memory_gene_pool) >= self.memory_capacity:
            self.memory_gene_pool.sort(key=lambda x: (x[1], -x[3]))
            self.memory_gene_pool.pop(0)
        self.memory_gene_pool.append((segment, utility, 1, self.total_attempts))
        print(f" [记忆基因池] 新增片段 {segment[:8]}... 效用值={utility:.2f}")

    # [v3.2.1] 调优：记忆池片段有效阈值从0.25降至0.20
    def _get_best_memory_segment(self):
        if not self.memory_gene_pool:
            return None
        best = max(self.memory_gene_pool, key=lambda x: x[1])
        return best[0] if best[1] > 0.20 else None

    def _build_chain_from_memory(self, segment, pressure):
        seg_len = len(segment)
        target_len = len(self.gene_chain)
        if seg_len >= target_len:
            return list(segment)[:target_len]
        else:
            remaining = target_len - seg_len
            if pressure > 7:
                fill = random.choices(['A', 'G'], k=remaining)
            elif pressure < 3:
                fill = random.choices(['C', 'T'], k=remaining)
            else:
                fill = random.choices(self.bases, k=remaining)
            return list(segment) + fill

    def _mutate_doubt_gene_chain(self):
        chain = self.doubt_gene_chain['primary']
        weights = {'A':0.1, 'G':0.2, 'C':0.2, 'T':0.3, 'X':0.2}
        strategy_to_base = {
            'aggressive': 'A',
            'conservative': 'C',
            'flexible': 'T',
            'explorative': 'X'
        }
        for strategy, base in strategy_to_base.items():
            util = self.doubt_strategy_utility.get(strategy, 0.5)
            if util > 0.55:
                weights[base] = weights.get(base, 0.1) * 1.5
        total = sum(weights.values())
        weights = {k: v/total for k,v in weights.items()}
        for _ in range(random.randint(2, 3)):
            pos = random.randint(0, len(chain)-1)
            new_base = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
            chain[pos] = new_base
        self.doubt_gene_chain['complementary'] = [self.complement_map[b] for b in chain]
        print(f" [怀疑基因链] 变异，新链前缀: {''.join(chain[:10])}...")

    # [v3.2.1] 调优：探索模式调用记忆池概率从0.3提升至0.5
    def _get_active_strand_for_move(self, board):
        pressure = self._calculate_board_pressure(board)
        if self.strategy_mode == "exploitative" and self.memory_gene_pool:
            best_segment = self._get_best_memory_segment()
            if best_segment:
                print(f" [活性链] 开发模式，使用记忆片段: {best_segment[:8]}...")
                return self._build_chain_from_memory(best_segment, pressure)
        if self.strategy_mode == "explorative" and random.random() < 0.5 and self.memory_gene_pool:
            best_segment = self._get_best_memory_segment()
            if best_segment:
                print(f" [活性链] 探索模式，尝试记忆片段: {best_segment[:8]}...")
                return self._build_chain_from_memory(best_segment, pressure)
        if self.strategy_mode == "explorative" or random.random() < self.exploration_rate:
            if random.random() < 0.5:
                print(f" [活性链] 探索模式，使用怀疑基因链")
                return self.doubt_gene_chain['primary']
            else:
                print(f" [活性链] 探索模式，使用互补链")
                return self.complementary_chain
        if self.strategy_mode == "balanced":
            if pressure > 0.7:
                return self.complementary_chain
            elif pressure < 0.3:
                return self.gene_chain
            else:
                return self.gene_chain if random.random() < 0.5 else self.complementary_chain
        return self.gene_chain

    def _calculate_board_pressure(self, board):
        analysis = board.get_pressure_analysis()
        if self.is_red:
            my_pressure = analysis['red_pressure']
            opp_pressure = analysis['black_pressure']
        else:
            my_pressure = analysis['black_pressure']
            opp_pressure = analysis['red_pressure']
        return max(0.1, min(0.9, 0.5 + (my_pressure - opp_pressure) * 0.3))

    def _evaluate_move_delta(self, board, move):
        temp = board.copy()
        piece, target = move
        tp = temp.get_piece_at(piece.position[0], piece.position[1])
        if tp is None or tp.is_red != piece.is_red:
            return 0.0
        temp.move_piece(tp, target)
        new_score = temp.evaluate_board_detailed(self.is_red)
        old_score = board.evaluate_board_detailed(self.is_red)
        return new_score - old_score

    def select_move_by_value_stochastic(self, board, legal_moves, gene_modifier=1.0):
        if not legal_moves:
            return None
        base_score = board.evaluate_board_detailed(self.is_red)
        scored_moves = []
        for move in legal_moves:
            delta = self._evaluate_move_delta(board, move)
            active = self._get_active_strand_for_move(board)
            gene_factor = 1.0
            if active:
                counts = Counter(active[:10])
                aggressive = counts.get('A',0) + counts.get('G',0)
                conservative = counts.get('C',0) + counts.get('T',0)
                if aggressive > conservative:
                    gene_factor = 1.2
                elif conservative > aggressive:
                    gene_factor = 0.8
            memory_bonus = 0.0
            if self.strategy_mode == "exploitative" and self.memory_gene_pool:
                memory_bonus = 20
            rule_bonus = self._apply_rule_bonus(board, move)
            delta = delta * gene_modifier * gene_factor + memory_bonus + rule_bonus
            scored_moves.append((delta, move))
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        top_moves = scored_moves[:5]
        deltas = [max(0.1, d[0]) for d in top_moves]
        total = sum(deltas)
        if total <= 0:
            return random.choice(top_moves)[1]
        probs = [d / total for d in deltas]
        chosen = random.choices([m[1] for m in top_moves], weights=probs)[0]
        return chosen

    def select_move_by_value_deterministic(self, board, legal_moves, gene_modifier=1.0):
        if not legal_moves:
            return None
        base_score = board.evaluate_board_detailed(self.is_red)
        best_move = None
        best_delta = -float('inf')
        for move in legal_moves:
            delta = self._evaluate_move_delta(board, move)
            active = self._get_active_strand_for_move(board)
            gene_factor = 1.0
            if active:
                counts = Counter(active[:10])
                if counts.get('A',0) + counts.get('G',0) > counts.get('C',0) + counts.get('T',0):
                    gene_factor = 1.2
                else:
                    gene_factor = 0.8
            delta *= gene_modifier * gene_factor
            if delta > best_delta:
                best_delta = delta
                best_move = move
        return best_move

    def _extract_features(self, board):
        features = {}
        red_elephants = sum(1 for p in board.pieces if not p.captured and p.is_red and p.piece_type == PieceType.ELEPHANT)
        black_elephants = sum(1 for p in board.pieces if not p.captured and not p.is_red and p.piece_type == PieceType.ELEPHANT)
        features['red_missing_elephant'] = red_elephants < 2
        features['black_missing_elephant'] = black_elephants < 2
        red_advisors = sum(1 for p in board.pieces if not p.captured and p.is_red and p.piece_type == PieceType.ADVISOR)
        black_advisors = sum(1 for p in board.pieces if not p.captured and not p.is_red and p.piece_type == PieceType.ADVISOR)
        features['red_missing_advisor'] = red_advisors < 2
        features['black_missing_advisor'] = black_advisors < 2
        for piece in board.pieces:
            if not piece.captured and piece.piece_type == PieceType.SOLDIER:
                if piece.is_red and piece.position[1] <= 4:
                    features['red_pawn_crossed'] = True
                if not piece.is_red and piece.position[1] >= 5:
                    features['black_pawn_crossed'] = True
        red_general = board.get_general(True)
        black_general = board.get_general(False)
        if red_general:
            x,y = red_general.position
            has_protection = False
            for dx,dy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                nx,ny = x+dx, y+dy
                piece = board.get_piece_at(nx,ny)
                if piece and piece.is_red and piece.piece_type in (PieceType.ADVISOR, PieceType.ELEPHANT):
                    has_protection = True
                    break
            features['red_general_exposed'] = not has_protection
        if black_general:
            x,y = black_general.position
            has_protection = False
            for dx,dy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                nx,ny = x+dx, y+dy
                piece = board.get_piece_at(nx,ny)
                if piece and not piece.is_red and piece.piece_type in (PieceType.ADVISOR, PieceType.ELEPHANT):
                    has_protection = True
                    break
            features['black_general_exposed'] = not has_protection
        features['is_my_turn'] = (self.is_red == board.current_player)
        return features

    def _categorize_move(self, move, board):
        piece, target = move
        src_piece = board.get_piece_at(piece.position[0], piece.position[1])
        if src_piece is None or src_piece.is_red != piece.is_red:
            return "other"
        if board.get_piece_at(target[0], target[1]) is not None:
            return "capture"
        temp = board.copy()
        tp = temp.get_piece_at(src_piece.position[0], src_piece.position[1])
        if tp is None:
            return "other"
        temp.move_piece(tp, target)
        if temp.is_in_check(not piece.is_red):
            return "check"
        if piece.piece_type == PieceType.GENERAL:
            return "general_move"
        if piece.piece_type in (PieceType.ADVISOR, PieceType.ELEPHANT):
            return "defense"
        if piece.piece_type in (PieceType.CHARIOT, PieceType.HORSE, PieceType.CANNON):
            if abs(target[0]-piece.position[0]) + abs(target[1]-piece.position[1]) > 2:
                return "long_move"
            else:
                return "short_move"
        if piece.piece_type == PieceType.SOLDIER:
            if piece.is_red and target[1] < piece.position[1] or not piece.is_red and target[1] > piece.position[1]:
                return "pawn_advance"
            else:
                return "pawn_side"
        return "other"

    def _learn_from_move(self, board, move, delta, is_immediate_success=None):
        if is_immediate_success is None:
            is_good_move = delta > 15
        else:
            is_good_move = is_immediate_success

        if board is not None:
            features = self._extract_features(board)
            move_type = self._categorize_move(move, board)
            for feat, value in features.items():
                if value:
                    key = f"{feat}->{move_type}"
                    self.rule_patterns[key]['total'] += 1
                    if is_good_move:
                        self.rule_patterns[key]['success'] += 1
                    else:
                        self.rule_patterns[key]['failure'] += 1
                    self.rule_patterns[key]['confidence'] = self.rule_patterns[key]['success'] / self.rule_patterns[key]['total']

        if is_good_move and delta > 15 and hasattr(self, '_last_active_strand'):
            strand = self._last_active_strand
            segment_size = 5
            explore_bonus = 0.2 if self.strategy_mode == "explorative" else 0.0
            for i in range(0, len(strand) - segment_size + 1):
                segment = tuple(strand[i:i+segment_size])
                utility = min(1.0, max(0.1, delta / 400)) + explore_bonus
                self._add_to_memory_pool(segment, utility)

        if self.last_doubt_strategy is not None:
            old_util = self.doubt_strategy_utility[self.last_doubt_strategy]
            feedback = 1.0 if delta > 0 else 0.0
            new_util = old_util * 0.9 + feedback * 0.1
            self.doubt_strategy_utility[self.last_doubt_strategy] = new_util
            self.last_doubt_strategy = None

    def _apply_rule_bonus(self, board, move):
        features = self._extract_features(board)
        move_type = self._categorize_move(move, board)
        bonus = 0.0
        for feat, value in features.items():
            if value:
                key = f"{feat}->{move_type}"
                if key in self.rule_patterns:
                    conf = self.rule_patterns[key]['confidence']
                    if conf > 0.6:
                        bonus += 30 * conf
        return bonus

    def _generate_doubt_signal(self, board, move):
        doubt_score = 0.0
        reasons = []
        if len(self.performance_history) >= 5:
            recent_fail = 1 - (sum(list(self.performance_history)[-5:]) / 5)
            doubt_score += recent_fail * 0.3
            if recent_fail > 0.5:
                reasons.append("近期胜率低")
        general = board.get_general(self.is_red)
        if general and general.pressure_score > 0.6:
            doubt_score += general.pressure_score * 0.4
            reasons.append("将帅危险")
        delta = self.last_decision_info.get('delta', 0) if self.last_decision_info['delta'] else 0
        if delta < 10:
            doubt_score += 0.2
            reasons.append("价值增量低")
        doubt_strand = self.doubt_gene_chain['primary'][:10]
        x_count = doubt_strand.count('X')
        t_count = doubt_strand.count('T')
        doubt_score += (x_count + t_count) * 0.05
        if x_count > 3:
            reasons.append("怀疑基因活跃")
        doubt_score += self.failure_streak * 0.1
        if self.failure_streak >= 2:
            reasons.append(f"连续失败{self.failure_streak}")
        doubt_score = min(0.9, doubt_score + random.uniform(-0.1, 0.1))
        should_doubt = random.random() < doubt_score
        reason_str = ", ".join(reasons) if reasons else "常规"
        return should_doubt, doubt_score, reason_str

    def _doubt_rethinking_semantic(self, initial_move, board, legal_moves):
        doubt_strand = self.doubt_gene_chain['primary'][:8]
        counts = Counter(doubt_strand)
        candidates = [initial_move]
        strategy_type = None
        if counts.get('A', 0) > 2:
            strategy_type = 'aggressive'
            worst_move = min(legal_moves, key=lambda m: self._evaluate_move_delta(board, m))
            if worst_move not in candidates:
                candidates.append(worst_move)
            print(f"  [怀疑推理] 激进反向: 选最劣走法")
        elif counts.get('C', 0) > 2:
            strategy_type = 'conservative'
            defensive = [m for m in legal_moves if self._categorize_move(m, board) not in ('capture', 'check')]
            if defensive:
                cand = random.choice(defensive)
                if cand not in candidates:
                    candidates.append(cand)
                print(f"  [怀疑推理] 保守反向: 选防守走法")
        elif counts.get('T', 0) > 2:
            strategy_type = 'flexible'
            random_move = random.choice(legal_moves)
            if random_move not in candidates:
                candidates.append(random_move)
            print(f"  [怀疑推理] 灵活反向: 随机走法")
        elif counts.get('X', 0) > 2:
            strategy_type = 'explorative'
            memory_segment = self._get_best_memory_segment()
            if memory_segment:
                cand = random.choice(legal_moves)
                candidates.append(cand)
            else:
                cand = random.choice(legal_moves)
                candidates.append(cand)
            print(f"  [怀疑推理] 探索反向: 记忆/随机走法")
        self.last_doubt_strategy = strategy_type
        best_move = initial_move
        best_score = self._evaluate_move_delta(board, initial_move)
        for move in candidates:
            delta = self._evaluate_move_delta(board, move)
            if delta > best_score:
                best_score = delta
                best_move = move
        return best_move

    def make_decision(self, board):
        self.total_attempts += 1
        self._update_strategy_mode()
        legal_moves = board.get_legal_moves(self.is_red)
        if not legal_moves:
            return None
        env_pressure = self._calculate_board_pressure(board)
        general = board.get_general(self.is_red)
        general_pressure = general.pressure_score if general else 0.0
        general_safe = general.pressure_score < 0.5 and general.threat_level < 0.3 if general else True
        self._last_active_strand = self._get_active_strand_for_move(board)
        initial_move = self.select_move_by_value_stochastic(board, legal_moves, gene_modifier=1.0)
        if not initial_move:
            initial_move = random.choice(legal_moves)
        sig = board.extract_situation_signature()
        pattern = self.long_term_memory['game_patterns'].get(sig)
        memory_hit = False
        memory_rate = 0.0
        if pattern and pattern['total'] > 5 and pattern['success_rate'] > 0.6 and random.random() < 0.8:
            best_move_hash = pattern.get('best_move_hash')
            if best_move_hash:
                for move in legal_moves:
                    if self._hash_move(move) == best_move_hash:
                        initial_move = move
                        memory_hit = True
                        memory_rate = pattern['success_rate']
                        print(f"  [记忆命中] 使用成功模式，成功率{pattern['success_rate']:.2f}")
                        break
        should_doubt, doubt_score, doubt_reason = self._generate_doubt_signal(board, initial_move)
        final_move = initial_move
        if should_doubt and len(legal_moves) > 1:
            doubted_move = self._doubt_rethinking_semantic(initial_move, board, legal_moves)
            if doubted_move and doubted_move != initial_move:
                final_move = doubted_move
                self.doubt_states.append(1)
                print(f"  [怀疑触发] 反向推理成功，改用新走法")
            else:
                self.doubt_states.append(0)
                print(f"  [怀疑触发] 反向推理未改进")
        else:
            self.doubt_states.append(0)
        delta = self._evaluate_move_delta(board, final_move)
        self.last_decision_info = {
            'initial_move': initial_move,
            'final_move': final_move,
            'doubt_triggered': should_doubt,
            'doubt_score': doubt_score,
            'doubt_reason': doubt_reason,
            'delta': delta,
            'memory_hit': memory_hit,
            'memory_success_rate': memory_rate,
            'strategy_mode': self.strategy_mode,
            'environment_pressure': env_pressure,
            'general_pressure': general_pressure,
            'general_safe': general_safe,
            'move_description': self._move_to_text(final_move, board),
            'timestamp': time.time()
        }
        board_snapshot = board.copy()
        self.game_moves.append((final_move, delta, board_snapshot))
        return final_move

    def apply_endgame_credit(self, is_win):
        if not self.game_moves:
            return
        print(f" [延迟信用] 本局共记录 {len(self.game_moves)} 步AI走法，结果: {'胜' if is_win else '负'}")
        for move, delta, board_snapshot in self.game_moves:
            adjusted_delta = delta * (1.2 if is_win else 0.8)
            is_good = is_win and delta > 5
            self._learn_from_move(board_snapshot, move, adjusted_delta, is_good)
        self.game_moves.clear()

    def _move_to_text(self, move, board):
        piece, target = move
        x, y = piece.position
        nx, ny = target
        piece_name = piece.get_symbol()
        from_pos = f"{chr(ord('a')+x)}{10-y}"
        to_pos = f"{chr(ord('a')+nx)}{10-ny}"
        return f"{piece_name}{from_pos}→{to_pos}"

    def _hash_move(self, move):
        piece, target = move
        return f"{piece.piece_type.value}{piece.is_red}{piece.position[0]}{piece.position[1]}{target[0]}{target[1]}"

    def mutate_gene_chain(self):
        if random.random() < 0.2:
            pos = random.randint(0, len(self.gene_chain)-1)
            new_base = random.choice([b for b in self.bases if b != self.gene_chain[pos]])
            self.gene_chain[pos] = new_base
            self.complementary_chain = [self.complement_map[b] for b in self.gene_chain]
            print("  [基因变异] 主链发生")

    # ================= 新增复盘学习方法 =================
    def learn_from_game_history(self, game_history, is_win):
        """从游戏历史记录中学习，分析最后几步的错误，将好的基因片段加入记忆池"""
        if is_win:  # 只学习失败的对局
            return
        if not game_history:
            return
        num_steps_to_review = 3  # 分析最后三步
        # 重建棋盘
        board = ChessBoard()
        steps = len(game_history)
        start_step = max(0, steps - num_steps_to_review)
        # 先执行前面的步，直到倒数第num_steps_to_review步之前
        for i in range(start_step):
            move = game_history[i]
            from_pos = move['from']
            to_pos = move['to']
            # 解析坐标
            from_x = ord(from_pos[0]) - ord('a')
            from_y_str = from_pos[1:]
            from_y = 10 - int(from_y_str)
            piece = board.get_piece_at(from_x, from_y)
            if piece is None:
                continue
            to_x = ord(to_pos[0]) - ord('a')
            to_y_str = to_pos[1:]
            to_y = 10 - int(to_y_str)
            board.move_piece(piece, (to_x, to_y))
        # 现在分析最后几步
        for i in range(start_step, steps):
            move = game_history[i]
            # 当前棋盘（走这一步之前）
            current_board = board.copy()
            from_pos = move['from']
            to_pos = move['to']
            from_x = ord(from_pos[0]) - ord('a')
            from_y_str = from_pos[1:]
            from_y = 10 - int(from_y_str)
            piece = current_board.get_piece_at(from_x, from_y)
            if piece is None:
                # 可能棋子已经被吃？但历史应该一致，跳过
                # 但需要推进board
                piece2 = board.get_piece_at(from_x, from_y)
                if piece2:
                    to_x = ord(to_pos[0]) - ord('a')
                    to_y_str = to_pos[1:]
                    to_y = 10 - int(to_y_str)
                    board.move_piece(piece2, (to_x, to_y))
                continue
            to_x = ord(to_pos[0]) - ord('a')
            to_y_str = to_pos[1:]
            to_y = 10 - int(to_y_str)
            actual_move = (piece, (to_x, to_y))
            # 检查这一步是否是本AI走的
            if move['player'] == ('红' if self.is_red else '黑'):
                # 评估实际走法的delta
                actual_delta = self._evaluate_move_delta(current_board, actual_move)
                # 获取当前局面所有合法走法
                legal_moves = current_board.get_legal_moves(self.is_red)
                best_move = None
                best_delta = -float('inf')
                for lm in legal_moves:
                    delta = self._evaluate_move_delta(current_board, lm)
                    if delta > best_delta:
                        best_delta = delta
                        best_move = lm
                # 如果最好走法明显优于实际走法
                if best_move and best_delta - actual_delta > 30:
                    # 获取当前局面下的活性链
                    active_strand = self._get_active_strand_for_move(current_board)
                    # 取前8个碱基作为片段
                    segment = tuple(active_strand[:8])
                    # 加入记忆池，效用设为1.0
                    self._add_to_memory_pool(segment, 1.0)
                    print(f" [复盘学习] 发现错误，将基因片段 {''.join(segment)} 加入记忆池")
                    # 也可以更新规则模式
                    features = self._extract_features(current_board)
                    move_type = self._categorize_move(best_move, current_board)
                    for feat, value in features.items():
                        if value:
                            key = f"{feat}->{move_type}"
                            self.rule_patterns[key]['total'] += 1
                            self.rule_patterns[key]['success'] += 1
                            self.rule_patterns[key]['confidence'] = self.rule_patterns[key]['success'] / self.rule_patterns[key]['total']
            # 执行实际移动，推进棋盘
            piece2 = board.get_piece_at(from_x, from_y)
            if piece2:
                board.move_piece(piece2, (to_x, to_y))
            else:
                print("复盘学习：无法找到棋子推进，可能历史记录不一致")

    def save_long_term_memory(self):
        os.makedirs(MEMORY_DIR, exist_ok=True)
        data = {
            'gene_chain': self.gene_chain,
            'complementary_chain': self.complementary_chain,
            'doubt_gene_chain': self.doubt_gene_chain,
            'memory_gene_pool': self.memory_gene_pool,
            'long_term_memory': self.long_term_memory,
            'gene_segment_utility': dict(self.gene_segment_utility),
            'success_count': self.success_count,
            'total_attempts': self.total_attempts,
            'doubt_effectiveness': self.doubt_effectiveness,
            'failure_memory': dict(self.failure_memory),
            'performance_history': list(self.performance_history),
            'rule_patterns': dict(self.rule_patterns),
            'doubt_strategy_utility': dict(self.doubt_strategy_utility),
        }
        with open(self.memory_file, 'wb') as f:
            pickle.dump(data, f)
        print(f" [记忆保存] 已保存至 {self.memory_file}")

    def load_long_term_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    data = pickle.load(f)
                self.gene_chain = data.get('gene_chain', self.gene_chain)
                self.complementary_chain = data.get('complementary_chain', self.complementary_chain)
                self.doubt_gene_chain = data.get('doubt_gene_chain', self.doubt_gene_chain)
                self.memory_gene_pool = data.get('memory_gene_pool', self.memory_gene_pool)
                self.long_term_memory = data.get('long_term_memory', self.long_term_memory)
                self.gene_segment_utility = defaultdict(float, data.get('gene_segment_utility', {}))
                self.success_count = data.get('success_count', 0)
                self.total_attempts = data.get('total_attempts', 0)
                self.doubt_effectiveness = data.get('doubt_effectiveness', 0.5)
                self.failure_memory = defaultdict(list, data.get('failure_memory', {}))
                hist = data.get('performance_history', [])
                self.performance_history = deque(hist[-20:], maxlen=20)
                rules = data.get('rule_patterns', {})
                self.rule_patterns = defaultdict(lambda: {'success':0,'failure':0,'total':0,'confidence':0.0})
                for k,v in rules.items():
                    self.rule_patterns[k] = v
                self.doubt_strategy_utility = defaultdict(float, data.get('doubt_strategy_utility', {}))
                print(f"  [长期记忆] 已加载 {self.memory_file}")
            except Exception as e:
                print(f"  [长期记忆] 加载失败: {e}")


# ============================================
# 5. 字体管理器
# ============================================
class FontManager:
    # 系统字体目录（根据操作系统设置）
    _FONT_DIRS = []
    if platform.system() == "Windows":
        _FONT_DIRS = [
            "C:/Windows/Fonts",
            os.path.expanduser("~/AppData/Local/Microsoft/Windows/Fonts"),
        ]
    elif platform.system() == "Darwin":  # macOS
        _FONT_DIRS = [
            "/System/Library/Fonts",
            "/Library/Fonts",
            os.path.expanduser("~/Library/Fonts"),
        ]
    else:  # Linux 及其他 Unix-like
        _FONT_DIRS = [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.fonts"),
            os.path.expanduser("~/.local/share/fonts"),
        ]

    # 中文字体文件名的关键词（用于初步筛选）
    _CHINESE_KEYWORDS = [
        "simhei", "simsun", "simfang", "simkai", "msyh",    # Windows
        "pingfang", "stheit", "hiragino",                   # macOS
        "noto", "wqy", "wenquanyi", "sourcehansans",        # Linux/跨平台
        "hei", "kai", "song", "fang", "yahei", "ming",      # 通用关键词
    ]

    _font_cache = {}  # 缓存已加载的字体 {size: font}

    @classmethod
    def get_font(cls, size: int):
        """获取指定大小的最优中文字体"""
        # 缓存命中
        if size in cls._font_cache:
            return cls._font_cache[size]

        # 动态搜索字体文件
        font_file = cls._find_chinese_font_file()
        if font_file:
            try:
                font = pygame.font.Font(font_file, size)
                cls._font_cache[size] = font
                return font
            except Exception:
                pass

        # 回退方案：Pygame 默认字体（可能不支持中文，但程序不崩溃）
        default_font = pygame.font.Font(None, size)
        cls._font_cache[size] = default_font
        return default_font

    @classmethod
    def _find_chinese_font_file(cls):
        """在系统字体目录中查找第一个可用的中文字体文件"""
        # 收集所有字体文件（ttf, ttc, otf）
        font_files = []
        for font_dir in cls._FONT_DIRS:
            if not os.path.isdir(font_dir):
                continue
            try:
                for file in os.listdir(font_dir):
                    if file.lower().endswith(('.ttf', '.ttc', '.otf')):
                        full_path = os.path.join(font_dir, file)
                        font_files.append(full_path)
            except PermissionError:
                continue

        # 按文件名排序，使结果更稳定（可选）
        font_files.sort()

        # 逐个测试字体是否支持中文
        for path in font_files:
            # 快速筛选：文件名包含中文字体关键词（提高效率）
            filename = os.path.basename(path).lower()
            if not any(keyword in filename for keyword in cls._CHINESE_KEYWORDS):
                continue

            try:
                font = pygame.font.Font(path, 20)  # 用小字号测试，减少开销
                # 渲染中文字符，若宽度太小则忽略
                surf = font.render("中", True, (0, 0, 0))
                if surf.get_width() > 10:  # 20号字体正常中文字符宽度应>10
                    return path
            except Exception:
                continue

        # 如果没有找到带关键词的字体，则尝试所有字体文件（兜底）
        for path in font_files:
            try:
                font = pygame.font.Font(path, 20)
                surf = font.render("中", True, (0, 0, 0))
                if surf.get_width() > 10:
                    return path
            except Exception:
                continue

        return None

    @classmethod
    def clear_cache(cls):
        cls._font_cache.clear()
# ============================================
# 6. 游戏GUI类（完整绘图实现）
# ============================================
class ChessGameGUI:
    def __init__(self, red_player: PlayerType = PlayerType.HUMAN, black_player: PlayerType = PlayerType.GEDA_AI):
        pygame.display.set_caption("GEDA象棋AI v3.2.2 - 复盘学习版")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = FontManager.get_font(28)
        self.large_font = FontManager.get_font(36)
        self.small_font = FontManager.get_font(22)
        self.tiny_font = FontManager.get_font(18)

        self.board = ChessBoard()
        self.game_state = GameState.PLAYING
        self.red_player = red_player
        self.black_player = black_player
        self.red_ai = GEDA_GeneSystem(is_red=True) if red_player == PlayerType.GEDA_AI else None
        self.black_ai = GEDA_GeneSystem(is_red=False) if black_player == PlayerType.GEDA_AI else None
        self.current_ai = self.red_ai if self.board.current_player else self.black_ai

        self.running = True
        self.selected_piece = None
        self.valid_moves = []
        self.last_move = None
        self.highlight_last_move = True
        self.game_paused = False
        self.ai_thinking = False
        self.ai_move_start_time = 0
        self.ai_thinking_time = 0
        self.move_count = 0
        self.red_captured = []
        self.black_captured = []
        self.game_history = []

        self.show_gene_map = False
        self.show_pressure_map = False
        self.show_ai_thinking = True
        self.show_piece_pressure = True

        os.makedirs(GAME_LOG_DIR, exist_ok=True)
        self.move_start_time = None

        print("=" * 70)
        print("GEDA象棋AI v3.2.2 - 复盘学习版")
        print("宏观正反馈 | 记忆基因池 | 怀疑语义化 | 规则学习 | 策略效用追踪 | 复盘学习")
        print("=" * 70)
        print(f"红方: {'人类' if red_player == PlayerType.HUMAN else 'GEDA AI'}")
        print(f"黑方: {'人类' if black_player == PlayerType.HUMAN else 'GEDA AI'}")
        print(f"对局日志将保存至: {GAME_LOG_DIR}")
        print("=" * 70)

    # ---------- 绘图方法（完整实现）----------
    def draw_board(self):
        self.screen.fill(BOARD_BROWN)
        for y in range(10):
            for x in range(9):
                rect = pygame.Rect(x * GRID_SIZE + MARGIN, y * GRID_SIZE + MARGIN, GRID_SIZE, GRID_SIZE)
                color = BOARD_LIGHT if (x + y) % 2 == 0 else BOARD_DARK
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, BLACK, rect, 1)
        # 楚河汉界
        river_rect = pygame.Rect(MARGIN, 4 * GRID_SIZE + MARGIN + GRID_SIZE // 2, 9 * GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(self.screen, (200, 200, 255, 128), river_rect)
        river_text = self.large_font.render("楚河          汉界", True, BLUE)
        self.screen.blit(river_text, (MARGIN + GRID_SIZE, 4 * GRID_SIZE + MARGIN + GRID_SIZE // 2 - 10))
        # 九宫标记
        for x in range(3, 6):
            for y in range(7, 10):
                rect = pygame.Rect(x * GRID_SIZE + MARGIN, y * GRID_SIZE + MARGIN, GRID_SIZE, GRID_SIZE)
                pygame.draw.rect(self.screen, (255, 200, 200, 100), rect)
        for x in range(3, 6):
            for y in range(0, 3):
                rect = pygame.Rect(x * GRID_SIZE + MARGIN, y * GRID_SIZE + MARGIN, GRID_SIZE, GRID_SIZE)
                pygame.draw.rect(self.screen, (200, 200, 255, 100), rect)
        # 九宫对角线
        lines = [
            ((3 * GRID_SIZE + MARGIN, 7 * GRID_SIZE + MARGIN), (5 * GRID_SIZE + MARGIN, 9 * GRID_SIZE + MARGIN)),
            ((5 * GRID_SIZE + MARGIN, 7 * GRID_SIZE + MARGIN), (3 * GRID_SIZE + MARGIN, 9 * GRID_SIZE + MARGIN)),
            ((3 * GRID_SIZE + MARGIN, 0 * GRID_SIZE + MARGIN), (5 * GRID_SIZE + MARGIN, 2 * GRID_SIZE + MARGIN)),
            ((5 * GRID_SIZE + MARGIN, 0 * GRID_SIZE + MARGIN), (3 * GRID_SIZE + MARGIN, 2 * GRID_SIZE + MARGIN))
        ]
        for s, e in lines:
            pygame.draw.line(self.screen, BLACK, s, e, 2)
        # 坐标标记
        for i in range(9):
            col_text = self.small_font.render(chr(ord('a') + i), True, BLACK)
            self.screen.blit(col_text, (i * GRID_SIZE + MARGIN + GRID_SIZE // 2 - 5, MARGIN - 20))
            self.screen.blit(col_text, (i * GRID_SIZE + MARGIN + GRID_SIZE // 2 - 5, 10 * GRID_SIZE + MARGIN + 5))
        for i in range(10):
            row_text = self.small_font.render(str(10 - i), True, BLACK)
            self.screen.blit(row_text, (MARGIN - 15, i * GRID_SIZE + MARGIN + GRID_SIZE // 2 - 8))
            self.screen.blit(row_text, (9 * GRID_SIZE + MARGIN + 5, i * GRID_SIZE + MARGIN + GRID_SIZE // 2 - 8))

    def draw_pieces(self):
        for piece in self.board.pieces:
            if piece.captured:
                continue
            x, y = piece.position
            sx = x * GRID_SIZE + MARGIN
            sy = y * GRID_SIZE + MARGIN
            center = (sx + GRID_SIZE // 2, sy + GRID_SIZE // 2)
            radius = GRID_SIZE // 2 - 4
            color = RED_PIECE_COLOR if piece.is_red else BLACK_PIECE_COLOR
            border = DARK_RED if piece.is_red else DARK_GRAY
            pygame.draw.circle(self.screen, color, center, radius)
            pygame.draw.circle(self.screen, border, center, radius, 2)
            symbol = piece.get_symbol()
            text_color = WHITE if piece.is_red else LIGHT_GRAY
            text_surface = (self.large_font if radius > 20 else self.font).render(symbol, True, text_color)
            self.screen.blit(text_surface, text_surface.get_rect(center=center))
        current_is_human = (self.board.current_player and self.red_player == PlayerType.HUMAN) or \
                           (not self.board.current_player and self.black_player == PlayerType.HUMAN)
        if current_is_human and self.selected_piece and not self.selected_piece.captured:
            x, y = self.selected_piece.position
            sx = x * GRID_SIZE + MARGIN
            sy = y * GRID_SIZE + MARGIN
            center = (sx + GRID_SIZE // 2, sy + GRID_SIZE // 2)
            pygame.draw.circle(self.screen, YELLOW, center, GRID_SIZE // 2 - 2, 3)

    def draw_valid_moves(self):
        if self.selected_piece and self.valid_moves:
            x, y = self.selected_piece.position
            cx = x * GRID_SIZE + MARGIN + GRID_SIZE // 2
            cy = y * GRID_SIZE + MARGIN + GRID_SIZE // 2
            for mx, my in self.valid_moves:
                tx = mx * GRID_SIZE + MARGIN + GRID_SIZE // 2
                ty = my * GRID_SIZE + MARGIN + GRID_SIZE // 2
                pygame.draw.line(self.screen, GREEN, (cx, cy), (tx, ty), 2)
                rect = pygame.Rect(tx - GRID_SIZE // 4, ty - GRID_SIZE // 4, GRID_SIZE // 2, GRID_SIZE // 2)
                pygame.draw.ellipse(self.screen, GREEN, rect, 2)
                if self.board.get_piece_at(mx, my):
                    pygame.draw.circle(self.screen, RED, (tx, ty), GRID_SIZE // 4, 2)

    def draw_info_panel(self):
        panel = pygame.Rect(INFO_PANEL_X, 10, INFO_PANEL_WIDTH, SCREEN_HEIGHT - 20)
        pygame.draw.rect(self.screen, DARK_GRAY, panel)
        pygame.draw.rect(self.screen, LIGHT_GRAY, panel, 2)
        y = 20

        title = self.large_font.render("GEDA决策监视器 v3.2.2", True, CYAN)
        self.screen.blit(title, (INFO_PANEL_X + 20, y))
        y += 40

        turn = "红方" if self.board.current_player else "黑方"
        turn_color = RED if self.board.current_player else BLACK_PIECE_COLOR
        turn_text = self.font.render(f"当前回合: {turn}", True, turn_color)
        self.screen.blit(turn_text, (INFO_PANEL_X + 20, y))
        y += 30

        state_text = self.font.render(f"状态: {self.game_state.name}", True, WHITE)
        self.screen.blit(state_text, (INFO_PANEL_X + 20, y))
        y += 25
        move_text = self.font.render(f"步数: {self.move_count}", True, YELLOW)
        self.screen.blit(move_text, (INFO_PANEL_X + 20, y))
        y += 30

        red_cnt = sum(1 for p in self.board.pieces if not p.captured and p.is_red)
        black_cnt = sum(1 for p in self.board.pieces if not p.captured and not p.is_red)
        piece_text = self.font.render(f"红{red_cnt} vs 黑{black_cnt}", True, WHITE)
        self.screen.blit(piece_text, (INFO_PANEL_X + 20, y))
        y += 40

        if self.show_piece_pressure:
            analysis = self.board.get_pressure_analysis()
            pressure_title = self.small_font.render("--- 压力分析 ---", True, ORANGE)
            self.screen.blit(pressure_title, (INFO_PANEL_X + 20, y))
            y += 22
            p_text = self.small_font.render(f"红方压力: {analysis['red_pressure']:.2f}", True, WHITE)
            self.screen.blit(p_text, (INFO_PANEL_X + 30, y))
            y += 20
            p_text2 = self.small_font.render(f"黑方压力: {analysis['black_pressure']:.2f}", True, WHITE)
            self.screen.blit(p_text2, (INFO_PANEL_X + 30, y))
            y += 20
            rg = analysis['general_status']['red']
            bg = analysis['general_status']['black']
            rg_text = self.small_font.render(f"红帅: {'安全' if rg['safe'] else '危险'} ({rg['pressure']:.2f})",
                                             True, GREEN if rg['safe'] else RED)
            self.screen.blit(rg_text, (INFO_PANEL_X + 30, y))
            y += 20
            bg_text = self.small_font.render(f"黑将: {'安全' if bg['safe'] else '危险'} ({bg['pressure']:.2f})",
                                             True, GREEN if bg['safe'] else RED)
            self.screen.blit(bg_text, (INFO_PANEL_X + 30, y))
            y += 20

        y += 10

        if self.current_ai and not self.ai_thinking:
            ai_title = self.font.render("--- 本次决策详情 ---", True, CYAN)
            self.screen.blit(ai_title, (INFO_PANEL_X + 20, y))
            y += 30

            info = self.current_ai.last_decision_info
            if info['final_move']:
                mode_color = GREEN if info['strategy_mode'] == 'exploitative' else \
                    RED if info['strategy_mode'] == 'explorative' else YELLOW
                mode_text = self.small_font.render(f"策略模式: {info['strategy_mode']}", True, mode_color)
                self.screen.blit(mode_text, (INFO_PANEL_X + 30, y))
                y += 22

                pcol = RED if info['environment_pressure'] > 0.7 else GREEN if info['environment_pressure'] < 0.3 else YELLOW
                env_text = self.small_font.render(f"环境压力: {info['environment_pressure']:.2f}", True, pcol)
                self.screen.blit(env_text, (INFO_PANEL_X + 30, y))
                y += 22

                gen_text = self.small_font.render(
                    f"将帅压力: {info['general_pressure']:.2f} ({'安全' if info['general_safe'] else '危险'})",
                    True, GREEN if info['general_safe'] else RED)
                self.screen.blit(gen_text, (INFO_PANEL_X + 30, y))
                y += 22

                doubt_color = RED if info['doubt_triggered'] else GREEN
                doubt_text = self.small_font.render(f"怀疑: {'是' if info['doubt_triggered'] else '否'}", True,
                                                    doubt_color)
                self.screen.blit(doubt_text, (INFO_PANEL_X + 30, y))
                y += 22
                if info['doubt_triggered']:
                    reason_text = self.tiny_font.render(f"原因: {info['doubt_reason']}", True, LIGHT_GRAY)
                    self.screen.blit(reason_text, (INFO_PANEL_X + 40, y))
                    y += 18

                if info['memory_hit']:
                    mem_text = self.small_font.render(f"记忆命中 (成功率 {info['memory_success_rate']:.0%})", True,
                                                      LIGHT_BLUE)
                    self.screen.blit(mem_text, (INFO_PANEL_X + 30, y))
                    y += 22

                move_desc = info['move_description'] if info['move_description'] else "无"
                move_text = self.small_font.render(f"最终走法: {move_desc}", True, YELLOW)
                self.screen.blit(move_text, (INFO_PANEL_X + 30, y))
                y += 22

                delta_color = GREEN if info['delta'] > 0 else RED
                delta_text = self.small_font.render(f"局面增值: {info['delta']:+.1f}", True, delta_color)
                self.screen.blit(delta_text, (INFO_PANEL_X + 30, y))
                y += 22

                rate = self.current_ai.success_count / max(self.current_ai.total_attempts, 1)
                rate_color = GREEN if rate > 0.6 else RED if rate < 0.3 else YELLOW
                rate_text = self.small_font.render(f"历史成功率: {rate:.1%}", True, rate_color)
                self.screen.blit(rate_text, (INFO_PANEL_X + 30, y))
                y += 22

                gene_len = len(self.current_ai.gene_chain)
                mem_cnt = len(self.current_ai.long_term_memory['game_patterns'])
                self.screen.blit(self.small_font.render(f"基因长度: {gene_len}", True, YELLOW),
                                 (INFO_PANEL_X + 30, y))
                y += 20
                self.screen.blit(self.small_font.render(f"记忆模式: {mem_cnt}", True, LIGHT_BLUE),
                                 (INFO_PANEL_X + 30, y))
                y += 20

                pool_size = len(self.current_ai.memory_gene_pool)
                self.screen.blit(self.small_font.render(f"记忆池片段: {pool_size}", True, ORANGE),
                                 (INFO_PANEL_X + 30, y))
                y += 20

                # 策略效用可视化
                su = self.current_ai.doubt_strategy_utility
                util_text = f"怀疑策略:A:{su.get('aggressive',0.5):.2f} C:{su.get('conservative',0.5):.2f} T:{su.get('flexible',0.5):.2f} X:{su.get('explorative',0.5):.2f}"
                self.screen.blit(self.tiny_font.render(util_text, True, LIGHT_BLUE), (INFO_PANEL_X + 30, y))
                y += 18
            else:
                self.screen.blit(self.small_font.render("等待AI首次决策...", True, WHITE), (INFO_PANEL_X + 30, y))
                y += 20
        elif self.ai_thinking:
            think = self.font.render("AI 思考中...", True, GREEN)
            self.screen.blit(think, (INFO_PANEL_X + 20, y))
            y += 30
            time_text = self.small_font.render(f"已用时: {self.ai_thinking_time:.1f}秒", True, YELLOW)
            self.screen.blit(time_text, (INFO_PANEL_X + 30, y))
            y += 20

        y += 10

        controls = [
            "鼠标: 选择/移动",
            "R: 重开  空格: 暂停",
            "Z: 悔棋(人类)",
            "S: 保存记忆  L: 加载",
            "T: 切换压力显示",
            "H: 高亮开关",
            "对局记录自动保存"
        ]
        ctrl_title = self.font.render("控制说明", True, WHITE)
        self.screen.blit(ctrl_title, (INFO_PANEL_X + 20, y))
        y += 25
        for c in controls:
            self.screen.blit(self.tiny_font.render(c, True, LIGHT_GRAY), (INFO_PANEL_X + 30, y))
            y += 18

    def draw_game_over(self):
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        if self.game_state == GameState.RED_WIN:
            msg = "红方获胜！"
            color = RED_PIECE_COLOR
        elif self.game_state == GameState.BLACK_WIN:
            msg = "黑方获胜！"
            color = BLACK_PIECE_COLOR
        else:
            msg = "平局！"
            color = YELLOW
        over_font = FontManager.get_font(72)
        over_text = over_font.render(msg, True, color)
        self.screen.blit(over_text, over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50)))
        restart = FontManager.get_font(36).render("按 R 键重新开始", True, WHITE)
        self.screen.blit(restart, restart.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30)))
        move_info = FontManager.get_font(36).render(f"总步数: {self.move_count}", True, YELLOW)
        self.screen.blit(move_info, move_info.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 80)))

    def draw_pause_screen(self):
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        pause = FontManager.get_font(72).render("游戏暂停", True, YELLOW)
        self.screen.blit(pause, pause.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50)))
        cont = FontManager.get_font(36).render("按 空格键 继续", True, WHITE)
        self.screen.blit(cont, cont.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30)))

    # ---------- 事件处理 ----------
    def handle_click(self, pos):
        if self.game_state != GameState.PLAYING or self.game_paused:
            return
        current_is_ai = (self.board.current_player and self.red_player == PlayerType.GEDA_AI) or \
                        (not self.board.current_player and self.black_player == PlayerType.GEDA_AI)
        if current_is_ai:
            return
        x, y = pos
        bx = (x - MARGIN) // GRID_SIZE
        by = (y - MARGIN) // GRID_SIZE
        if not (0 <= bx < 9 and 0 <= by < 10):
            return
        clicked = self.board.get_piece_at(bx, by)
        if not self.selected_piece:
            if clicked and clicked.is_red == self.board.current_player:
                self.selected_piece = clicked
                self.valid_moves = [move for move in clicked.get_moves(self.board)
                                    if self.board.is_legal_move(clicked, move)]
                self.move_start_time = time.time()
        else:
            if clicked and clicked.is_red == self.board.current_player:
                self.selected_piece = clicked
                self.valid_moves = [move for move in clicked.get_moves(self.board)
                                    if self.board.is_legal_move(clicked, move)]
                self.move_start_time = time.time()
            elif (bx, by) in self.valid_moves:
                self.execute_move(self.selected_piece, (bx, by))
            else:
                self.selected_piece = None
                self.valid_moves = []

    def execute_move(self, piece, target):
        moving_player_is_red = self.board.current_player

        score_before = self.board.evaluate_board_detailed(moving_player_is_red)
        fen_before = self.board.extract_situation_signature()
        from_pos = f"{chr(ord('a') + piece.position[0])}{10 - piece.position[1]}"
        to_pos = f"{chr(ord('a') + target[0])}{10 - target[1]}"

        if not self.board.move_piece(piece, target):
            return False

        score_after = self.board.evaluate_board_detailed(moving_player_is_red)
        fen_after = self.board.extract_situation_signature()
        delta = score_after - score_before

        is_check = self.board.is_in_check(not moving_player_is_red)
        is_mate = False
        if is_check:
            opp_legal = self.board.get_legal_moves(not moving_player_is_red)
            if not opp_legal:
                is_mate = True

        captured = None
        if len(self.board.move_history) > 0:
            last_move = self.board.move_history[-1]
            captured = last_move['captured']

        think_time = 0.0
        if self.move_start_time is not None:
            think_time = time.time() - self.move_start_time
            self.move_start_time = None

        move_record = {
            'move_number': self.move_count + 1,
            'player': '红' if moving_player_is_red else '黑',
            'piece_symbol': piece.get_symbol(),
            'from': from_pos,
            'to': to_pos,
            'fen_before': fen_before,
            'fen_after': fen_after,
            'score_before': round(score_before, 2),
            'score_after': round(score_after, 2),
            'delta': round(delta, 2),
            'is_check': is_check,
            'is_mate': is_mate,
            'is_capture': captured is not None,
            'think_time': round(think_time, 2),
            'timestamp': time.time()
        }

        current_is_ai = (moving_player_is_red and self.red_player == PlayerType.GEDA_AI) or \
                        (not moving_player_is_red and self.black_player == PlayerType.GEDA_AI)
        if current_is_ai:
            ai = self.red_ai if moving_player_is_red else self.black_ai
            if ai:
                move_record['ai_decision'] = ai.last_decision_info.copy()
                ai._learn_from_move(self.board, (piece, target), delta)

        self.game_state = self.board.game_state
        self.last_move = (piece.position, target, piece)
        self.move_count += 1

        if captured:
            if captured.is_red:
                self.red_captured.append(captured)
            else:
                self.black_captured.append(captured)

        self.game_history.append(move_record)

        self.selected_piece = None
        self.valid_moves = []

        if self.game_state != GameState.PLAYING:
            self.on_game_end()

        return True

    def ai_make_move(self):
        if self.game_state != GameState.PLAYING or self.game_paused:
            return
        legal_moves = self.board.get_legal_moves(self.board.current_player)
        if not legal_moves:
            if self.board.current_player:
                self.game_state = GameState.BLACK_WIN
            else:
                self.game_state = GameState.RED_WIN
            self.on_game_end()
            return
        self.current_ai = self.red_ai if self.board.current_player else self.black_ai
        if not self.current_ai:
            return
        self.ai_thinking = True
        self.ai_move_start_time = time.time()
        move = self.current_ai.make_decision(self.board)
        self.ai_thinking_time = time.time() - self.ai_move_start_time
        self.ai_thinking = False
        if move:
            piece, target = move
            self.execute_move(piece, target)
            if random.random() < 0.2:
                self.current_ai.mutate_gene_chain()
            if random.random() < 0.1:
                self.current_ai._mutate_doubt_gene_chain()
        else:
            if self.board.current_player:
                self.game_state = GameState.BLACK_WIN
            else:
                self.game_state = GameState.RED_WIN
            self.on_game_end()

    def undo_move(self):
        if self.move_count == 0 or self.game_state != GameState.PLAYING:
            return
        current_is_ai = (self.board.current_player and self.red_player == PlayerType.GEDA_AI) or \
                        (not self.board.current_player and self.black_player == PlayerType.GEDA_AI)
        if current_is_ai and self.move_count >= 2:
            self.board.undo_move()
            self.move_count -= 1
            if self.game_history:
                self.game_history.pop()
            self.board.undo_move()
            self.move_count -= 1
            if self.game_history:
                self.game_history.pop()
            print("悔棋两步")
        elif not current_is_ai:
            self.board.undo_move()
            self.move_count -= 1
            if self.game_history:
                self.game_history.pop()
            print("悔棋一步")
        self.game_state = self.board.game_state
        self.selected_piece = None
        self.valid_moves = []

    def restart_game(self):
        # 重置对局，不应用延迟信用（无效局）
        # 清空AI的game_moves，避免未完成对局被学习
        if self.red_ai:
            self.red_ai.game_moves.clear()
        if self.black_ai:
            self.black_ai.game_moves.clear()

        # 可选：保存当前AI记忆（但不清除）
        if self.red_ai:
            self.red_ai.save_long_term_memory()
        if self.black_ai:
            self.black_ai.save_long_term_memory()

        # 保存未完成对局的日志（可选）
        if self.game_history:
            self.save_game_record()

        # 重置棋盘和游戏状态，但保留现有AI对象
        self.board = ChessBoard()
        self.game_state = GameState.PLAYING
        self.selected_piece = None
        self.valid_moves = []
        self.last_move = None
        self.move_count = 0
        self.red_captured = []
        self.black_captured = []
        self.game_history = []
        self.ai_thinking = False
        self.current_ai = self.red_ai if self.board.current_player else self.black_ai
        self.move_start_time = None
        print("游戏重新开始")

    def on_game_end(self):
        print(f"游戏结束！{self.game_state.name}")
        winner_is_red = (self.game_state == GameState.RED_WIN)

        # 复盘学习：输的一方学习错误
        if self.red_player == PlayerType.GEDA_AI and self.red_ai:
            if not winner_is_red:  # 红方输了
                self.red_ai.learn_from_game_history(self.game_history, False)
        if self.black_player == PlayerType.GEDA_AI and self.black_ai:
            if winner_is_red:  # 黑方输了
                self.black_ai.learn_from_game_history(self.game_history, False)

        if self.red_ai:
            self.red_ai.apply_endgame_credit(winner_is_red)
        if self.black_ai:
            self.black_ai.apply_endgame_credit(not winner_is_red)

        self.save_game_record()
        if self.red_ai:
            self.red_ai.save_long_term_memory()
        if self.black_ai:
            self.black_ai.save_long_term_memory()

    def save_game_record(self):
        if not self.game_history:
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        filename = f"game_{timestamp}.txt"
        filepath = os.path.join(GAME_LOG_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("                          GEDA 象棋AI v3.2.2 对局记录\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"对局时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"红方: {'人类' if self.red_player == PlayerType.HUMAN else 'GEDA AI'}\n")
            f.write(f"黑方: {'人类' if self.black_player == PlayerType.HUMAN else 'GEDA AI'}\n")
            f.write(f"结果: {self.game_state.name}\n")
            f.write(f"总步数: {self.move_count}\n\n")

            f.write("【对局过程】\n")
            f.write("-" * 60 + "\n")
            f.write("步数 | 走棋方 | 棋子 |  从   |  到   | 评分变化 | 吃子 | 将军 | 将死 | 思考时间\n")
            f.write("-" * 60 + "\n")
            for move in self.game_history:
                delta_str = f"{move['delta']:+7.2f}"
                capture_str = "✓" if move['is_capture'] else " "
                check_str = "✓" if move['is_check'] else " "
                mate_str = "✓" if move['is_mate'] else " "
                time_str = f"{move['think_time']:5.2f}s" if move['think_time'] > 0 else "     "
                f.write(
                    f"{move['move_number']:3d}  |  {move['player']}    |  {move['piece_symbol']}  "
                    f"| {move['from']:>3} | {move['to']:>3} |  {delta_str:>6}  "
                    f"|  {capture_str}   |  {check_str}   |  {mate_str}   | {time_str}\n")

            f.write("\n\n")
            f.write("=" * 80 + "\n")
            f.write("【每一步的完整监控数据】\n")
            f.write("=" * 80 + "\n\n")

            for move in self.game_history:
                f.write(
                    f"第{move['move_number']}步: {move['player']} {move['piece_symbol']} {move['from']} → {move['to']}\n")
                f.write(f"  移动前评分: {move['score_before']:8.2f}  (红方视角)\n")
                f.write(f"  移动后评分: {move['score_after']:8.2f}  (红方视角)\n")
                f.write(f"  评分增量:   {move['delta']:+8.2f}\n")
                f.write(f"  局面特征码: {move['fen_before']} → {move['fen_after']}\n")
                f.write(
                    f"  吃子: {'是' if move['is_capture'] else '否'}   将军: {'是' if move['is_check'] else '否'}   将死: {'是' if move['is_mate'] else '否'}\n")
                f.write(f"  思考时间: {move['think_time']:.2f} 秒\n")

                if 'ai_decision' in move:
                    ai = move['ai_decision']
                    f.write("  【AI决策监控】\n")
                    f.write(f"    策略模式: {ai['strategy_mode']}\n")
                    f.write(f"    环境压力: {ai['environment_pressure']:.2f}\n")
                    f.write(f"    将帅压力: {ai['general_pressure']:.2f} ({'安全' if ai['general_safe'] else '危险'})\n")
                    f.write(f"    怀疑触发: {'是' if ai['doubt_triggered'] else '否'}")
                    if ai['doubt_triggered']:
                        f.write(f" (得分: {ai['doubt_score']:.2f}, 原因: {ai['doubt_reason']})\n")
                    else:
                        f.write("\n")
                    f.write(f"    记忆命中: {'是' if ai['memory_hit'] else '否'}")
                    if ai['memory_hit']:
                        f.write(f" (成功率: {ai['memory_success_rate']:.0%})\n")
                    else:
                        f.write("\n")
                    f.write(f"    最终走法: {ai['move_description']}\n")
                    f.write(f"    局面增值: {ai['delta']:+.1f}\n")

                f.write("\n" + "-" * 60 + "\n")

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("记录结束\n")
            f.write("=" * 80 + "\n")

        print(f"对局记录已保存至: {filepath}")

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.handle_click(event.pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.restart_game()
                elif event.key == pygame.K_SPACE:
                    self.game_paused = not self.game_paused
                elif event.key == pygame.K_h:
                    self.highlight_last_move = not self.highlight_last_move
                elif event.key == pygame.K_t:
                    self.show_piece_pressure = not self.show_piece_pressure
                elif event.key == pygame.K_z:
                    if not self.ai_thinking:
                        self.undo_move()
                elif event.key == pygame.K_s:
                    if self.red_ai:
                        self.red_ai.save_long_term_memory()
                    if self.black_ai:
                        self.black_ai.save_long_term_memory()
                    print("手动保存AI记忆")
                elif event.key == pygame.K_l:
                    if self.red_ai:
                        self.red_ai.load_long_term_memory()
                    if self.black_ai:
                        self.black_ai.load_long_term_memory()
                    print("手动加载AI记忆")

    def update(self):
        if self.game_paused or self.game_state != GameState.PLAYING:
            return
        legal_moves = self.board.get_legal_moves(self.board.current_player)
        if not legal_moves:
            if self.board.current_player:
                self.game_state = GameState.BLACK_WIN
            else:
                self.game_state = GameState.RED_WIN
            self.on_game_end()
            return
        current_is_ai = (self.board.current_player and self.red_player == PlayerType.GEDA_AI) or \
                        (not self.board.current_player and self.black_player == PlayerType.GEDA_AI)
        if current_is_ai and not self.ai_thinking:
            if not hasattr(self, 'ai_delay') or time.time() - self.ai_delay > 0.5:
                self.ai_make_move()
                self.ai_delay = time.time()

    def draw(self):
        self.draw_board()
        self.draw_pieces()
        self.draw_valid_moves()
        self.draw_info_panel()
        if self.game_state != GameState.PLAYING:
            self.draw_game_over()
        if self.game_paused:
            self.draw_pause_screen()

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()
        if self.red_ai:
            self.red_ai.save_long_term_memory()
        if self.black_ai:
            self.black_ai.save_long_term_memory()
        print("游戏结束，再见！")


# ============================================
# 7. 主程序
# ============================================
def main():
    os.makedirs(MEMORY_DIR, exist_ok=True)
    game = ChessGameGUI(
        red_player=PlayerType.HUMAN,
        black_player=PlayerType.GEDA_AI
    )
    game.run()


if __name__ == "__main__":
    main()