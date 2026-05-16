# ============================================
# GEDA 五子棋AI - 带防守评估的纯基因系统
# 功能：记忆池 + 怀疑语义化 + 动态位置评分 + 防守价值（基于模拟对手）
# 棋盘：15x15，黑白双方，黑先
# 按键：R重开，空格暂停，Z悔棋，S/L保存/加载记忆，H高亮最后一步，P切换动态评分显示
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
import hashlib
import platform

# 初始化pygame
pygame.init()

# ============================================
# 1. 全局常量与配置
# ============================================

BOARD_SIZE = 15          # 五子棋盘大小
GRID_SIZE = 40           # 每个格子大小
BOARD_WIDTH = BOARD_SIZE * GRID_SIZE
BOARD_HEIGHT = BOARD_SIZE * GRID_SIZE
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

# 棋子颜色
BLACK_PIECE_COLOR = (30, 30, 30)
WHITE_PIECE_COLOR = (240, 240, 240)

# GEDA基因系统常量
GENE_BASES = ['A', 'G', 'C', 'T', 'X']
GENE_COMPLEMENTS = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'X': 'X'}

# 长期记忆文件路径
MEMORY_DIR = "geda_memory"
BLACK_MEMORY_FILE = os.path.join(MEMORY_DIR, "black_ai_memory_v3.pkl")   # 黑方记忆
WHITE_MEMORY_FILE = os.path.join(MEMORY_DIR, "white_ai_memory_v3.pkl")  # 白方记忆
GAME_LOG_DIR = os.path.join(MEMORY_DIR, "game_logs")

# 游戏状态
class GameState(Enum):
    PLAYING = 1
    BLACK_WIN = 2   # 黑胜
    WHITE_WIN = 3   # 白胜
    DRAW = 4

# 玩家类型
class PlayerType(Enum):
    HUMAN = 1
    GEDA_AI = 2


# ============================================
# 2. 棋子类
# ============================================
class ChessPiece:
    def __init__(self, is_black: bool, position: Tuple[int, int]):
        self.is_black = is_black
        self.position = position
        self.captured = False

    def get_symbol(self):
        return "●"

    def __repr__(self):
        color = "黑" if self.is_black else "白"
        return f"{color}棋({self.position})"


# ============================================
# 3. 棋盘类（五子棋实现）
# ============================================
class ChessBoard:
    def __init__(self):
        self.board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.pieces = []
        self.current_player = True  # True表示黑方先手
        self.game_state = GameState.PLAYING
        self.move_history = []      # 记录每一步的棋子对象

    def setup_board(self):
        """清空棋盘"""
        self.board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.pieces = []
        self.current_player = True
        self.game_state = GameState.PLAYING
        self.move_history = []

    def get_piece_at(self, x: int, y: int) -> Optional[ChessPiece]:
        if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
            return self.board[y][x]
        return None

    def place_piece(self, x: int, y: int) -> bool:
        """在指定位置落子，成功返回True"""
        if self.get_piece_at(x, y) is not None:
            return False
        piece = ChessPiece(self.current_player, (x, y))
        self.board[y][x] = piece
        self.pieces.append(piece)
        self.move_history.append(piece)
        self.current_player = not self.current_player
        self.check_game_state(x, y)
        return True

    def undo_move(self) -> bool:
        """悔棋一步"""
        if not self.move_history:
            return False
        last = self.move_history.pop()
        x, y = last.position
        self.board[y][x] = None
        self.pieces.remove(last)
        self.current_player = not self.current_player
        self.game_state = GameState.PLAYING
        return True

    def check_game_state(self, last_x: int, last_y: int):
        """检查最后落子是否导致胜利或平局"""
        piece = self.get_piece_at(last_x, last_y)
        if piece is None:
            return
        color = piece.is_black
        # 四个方向：水平、垂直、两个对角线
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in directions:
            count = 1
            # 正方向
            x, y = last_x + dx, last_y + dy
            while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                p = self.get_piece_at(x, y)
                if p and p.is_black == color:
                    count += 1
                    x += dx
                    y += dy
                else:
                    break
            # 负方向
            x, y = last_x - dx, last_y - dy
            while 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                p = self.get_piece_at(x, y)
                if p and p.is_black == color:
                    count += 1
                    x -= dx
                    y -= dy
                else:
                    break
            if count >= 5:
                self.game_state = GameState.BLACK_WIN if color else GameState.WHITE_WIN
                return
        # 检查是否满盘
        if len(self.pieces) == BOARD_SIZE * BOARD_SIZE:
            self.game_state = GameState.DRAW

    def get_legal_moves(self, player_black: bool) -> List[Tuple[Optional[ChessPiece], Tuple[int, int]]]:
        """返回所有空位，每个元素为(dummy_piece, (x,y))，dummy_piece用于传递玩家颜色"""
        moves = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board[y][x] is None:
                    dummy = ChessPiece(player_black, (-1, -1))
                    moves.append((dummy, (x, y)))
        return moves

    def evaluate_board_detailed(self, for_black: bool) -> float:
        """
        评估当前局面，返回对指定方有利的分数（正值有利）
        基于每个棋子的连续长度和位置权重
        """
        black_score = 0.0
        white_score = 0.0
        center = BOARD_SIZE // 2
        for piece in self.pieces:
            x, y = piece.position
            # 位置权重：距离中心越近越高
            dist = (x - center) ** 2 + (y - center) ** 2
            pos_weight = 1.0 / (1.0 + dist / 50.0)
            # 四个方向
            directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
            for dx, dy in directions:
                count = 1
                # 正方向
                nx, ny = x + dx, y + dy
                while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                    p = self.board[ny][nx]
                    if p and p.is_black == piece.is_black:
                        count += 1
                        nx += dx
                        ny += dy
                    else:
                        break
                # 负方向
                nx, ny = x - dx, y - dy
                while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                    p = self.board[ny][nx]
                    if p and p.is_black == piece.is_black:
                        count += 1
                        nx -= dx
                        ny -= dy
                    else:
                        break
                line_score = count ** 4 * pos_weight
                if piece.is_black:
                    black_score += line_score
                else:
                    white_score += line_score
        diff = black_score - white_score
        return diff if for_black else -diff

    def extract_situation_signature(self) -> str:
        """生成当前局面的哈希签名，用于记忆匹配"""
        sig_parts = []
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                piece = self.board[y][x]
                if piece:
                    sig_parts.append(f"{'B' if piece.is_black else 'W'}{x}{y}")
        sig_parts.sort()
        sig_parts.append(f"turn{'B' if self.current_player else 'W'}")
        full = "".join(sig_parts)
        return hashlib.md5(full.encode()).hexdigest()[:16]

    def copy(self) -> 'ChessBoard':
        """深拷贝棋盘"""
        new_board = ChessBoard()
        new_board.board = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        new_board.pieces = []
        for piece in self.pieces:
            new_piece = ChessPiece(piece.is_black, piece.position)
            new_piece.captured = piece.captured
            new_board.pieces.append(new_piece)
            new_board.board[piece.position[1]][piece.position[0]] = new_piece
        new_board.current_player = self.current_player
        new_board.game_state = self.game_state
        new_board.move_history = self.move_history.copy()
        return new_board


# ============================================
# 4. GEDA基因系统（带防守评估）
# ============================================
class GEDA_GeneSystem:
    def __init__(self, is_black: bool, memory_file: str = None):
        self.is_black = is_black
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

        self.rule_patterns = defaultdict(lambda: {'success':0,'failure':0,'total':0,'confidence':0.0})
        self.pressure_correlations = Counter()

        self.last_decision_info = {
            'initial_move': None,
            'final_move': None,
            'doubt_triggered': False,
            'doubt_score': 0.0,
            'doubt_reason': '',
            'delta': 0.0,
            'attack_delta': 0.0,        # 新增：进攻得分
            'defense_delta': 0.0,        # 新增：防守得分
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

        self.memory_file = memory_file if memory_file else (BLACK_MEMORY_FILE if is_black else WHITE_MEMORY_FILE)

        self.success_count = 0
        self.total_attempts = 0
        self.recent_success_rate = 0.0

        # 动态位置评分表（由基因变异调整）
        self.position_scores = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

        # 防守权重（初始0.8，可由变异微调）
        self.defense_weight = 0.8

        self.load_long_term_memory()
        print(f"GEDA基因系统初始化 - {'黑方' if is_black else '白方'}，基因长度: {len(self.gene_chain)}")

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

    def _get_active_strand_for_move(self, board):
        if self.strategy_mode == "exploitative" and self.memory_gene_pool:
            best_segment = self._get_best_memory_segment()
            if best_segment:
                print(f" [活性链] 开发模式，使用记忆片段: {best_segment[:8]}...")
                return self._build_chain_from_memory(best_segment, 5)
        if self.strategy_mode == "explorative" and random.random() < 0.5 and self.memory_gene_pool:
            best_segment = self._get_best_memory_segment()
            if best_segment:
                print(f" [活性链] 探索模式，尝试记忆片段: {best_segment[:8]}...")
                return self._build_chain_from_memory(best_segment, 5)
        if self.strategy_mode == "explorative" or random.random() < self.exploration_rate:
            if random.random() < 0.5:
                print(f" [活性链] 探索模式，使用怀疑基因链")
                return self.doubt_gene_chain['primary']
            else:
                print(f" [活性链] 探索模式，使用互补链")
                return self.complementary_chain
        if self.strategy_mode == "balanced":
            return random.choice([self.gene_chain, self.complementary_chain])
        return self.gene_chain

    def _calculate_board_pressure(self, board):
        return 0.5

    def _evaluate_move_delta(self, board, move):
        """
        计算落子的综合价值 = 进攻价值 + 防守权重 * 防守价值
        进攻价值：自己落子后自己视角的得分增量
        防守价值：对手落子后对手视角的得分增量（我方应尽量避免对手得分，所以防守价值越高越好）
        """
        dummy, target = move
        x, y = target
        if board.get_piece_at(x, y) is not None:
            return 0.0

        # 进攻价值
        temp_attack = board.copy()
        new_piece = ChessPiece(dummy.is_black, (x, y))
        temp_attack.board[y][x] = new_piece
        temp_attack.pieces.append(new_piece)
        new_score = temp_attack.evaluate_board_detailed(dummy.is_black)
        old_score = board.evaluate_board_detailed(dummy.is_black)
        attack_delta = new_score - old_score

        # 防守价值：假设对手落子于此，计算对手视角的得分增量
        temp_defense = board.copy()
        opp_piece = ChessPiece(not dummy.is_black, (x, y))
        temp_defense.board[y][x] = opp_piece
        temp_defense.pieces.append(opp_piece)
        opp_new_score = temp_defense.evaluate_board_detailed(not dummy.is_black)
        opp_old_score = board.evaluate_board_detailed(not dummy.is_black)
        defense_delta = opp_new_score - opp_old_score  # 对手得分增加量

        # 综合价值：进攻 + 防守权重 * 防守价值（防守价值对我方有利，所以加上）
        total_delta = attack_delta + self.defense_weight * defense_delta

        # 加上动态位置评分
        if board.get_piece_at(x, y) is None:
            total_delta += self.position_scores[y, x]

        # 保存详细得分供显示
        self.last_decision_info['attack_delta'] = attack_delta
        self.last_decision_info['defense_delta'] = defense_delta

        return total_delta

    def select_move_by_value_stochastic(self, board, legal_moves, gene_modifier=1.0):
        if not legal_moves:
            return None
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
            rule_bonus = 0
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
        return {}

    def _categorize_move(self, move, board):
        return "move"

    def _learn_from_move(self, board, move, delta, is_immediate_success=None):
        if is_immediate_success is None:
            is_good_move = delta > 15
        else:
            is_good_move = is_immediate_success

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
        return 0

    def _generate_doubt_signal(self, board, move):
        doubt_score = 0.0
        reasons = []
        if len(self.performance_history) >= 5:
            recent_fail = 1 - (sum(list(self.performance_history)[-5:]) / 5)
            doubt_score += recent_fail * 0.3
            if recent_fail > 0.5:
                reasons.append("近期胜率低")
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
            random_move = random.choice(legal_moves)
            if random_move not in candidates:
                candidates.append(random_move)
            print(f"  [怀疑推理] 保守反向: 随机走法")
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
        legal_moves = board.get_legal_moves(self.is_black)
        if not legal_moves:
            return None
        env_pressure = self._calculate_board_pressure(board)

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
        self.last_decision_info.update({
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
            'general_pressure': 0.0,
            'general_safe': True,
            'move_description': self._move_to_text(final_move, board),
            'timestamp': time.time()
        })
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
        dummy, target = move
        x, y = target
        color = "黑" if dummy.is_black else "白"
        return f"{color}棋({x},{y})"

    def _hash_move(self, move):
        dummy, target = move
        return f"{'B' if dummy.is_black else 'W'}{target[0]}{target[1]}"

    def mutate_gene_chain(self):
        if random.random() < 0.2:
            pos = random.randint(0, len(self.gene_chain)-1)
            new_base = random.choice([b for b in self.bases if b != self.gene_chain[pos]])
            self.gene_chain[pos] = new_base
            self.complementary_chain = [self.complement_map[b] for b in self.gene_chain]
            print("  [基因变异] 主链发生")
        if random.random() < 0.3:
            num_changes = random.randint(1, 5)
            for _ in range(num_changes):
                x = random.randint(0, BOARD_SIZE-1)
                y = random.randint(0, BOARD_SIZE-1)
                change = random.uniform(-5, 5)
                self.position_scores[y, x] = np.clip(self.position_scores[y, x] + change, -30, 30)
            print(f"  [位置评分变异] 调整了{num_changes}个位置的评分")
        # 防守权重变异
        if random.random() < 0.1:
            change = random.uniform(-0.2, 0.2)
            self.defense_weight = np.clip(self.defense_weight + change, 0.2, 1.5)
            print(f"  [防守权重变异] 新值={self.defense_weight:.2f}")

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
            'position_scores': self.position_scores.tolist(),
            'defense_weight': self.defense_weight,
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
                if 'position_scores' in data:
                    self.position_scores = np.array(data['position_scores'], dtype=np.float32)
                if 'defense_weight' in data:
                    self.defense_weight = data['defense_weight']
                print(f"  [长期记忆] 已加载 {self.memory_file}")
            except Exception as e:
                print(f"  [长期记忆] 加载失败: {e}")


# ============================================
# 5. 字体管理器（优化版 - 自动选择有效中文字体）
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
# 6. 游戏GUI类（增加进攻/防守得分显示）
# ============================================
class GomokuGameGUI:
    def __init__(self, black_player: PlayerType = PlayerType.HUMAN, white_player: PlayerType = PlayerType.GEDA_AI):
        pygame.display.set_caption("GEDA五子棋AI - 带防守评估")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = FontManager.get_font(28)
        self.large_font = FontManager.get_font(36)
        self.small_font = FontManager.get_font(22)
        self.tiny_font = FontManager.get_font(18)

        self.board = ChessBoard()
        self.game_state = GameState.PLAYING
        self.black_player = black_player
        self.white_player = white_player
        self.black_ai = GEDA_GeneSystem(is_black=True) if black_player == PlayerType.GEDA_AI else None
        self.white_ai = GEDA_GeneSystem(is_black=False) if white_player == PlayerType.GEDA_AI else None
        self.current_ai = self.black_ai if self.board.current_player else self.white_ai

        self.running = True
        self.selected_piece = None
        self.last_move = None
        self.highlight_last_move = True
        self.game_paused = False
        self.ai_thinking = False
        self.ai_move_start_time = 0
        self.ai_thinking_time = 0
        self.move_count = 0
        self.game_history = []

        self.show_position_scores = False

        os.makedirs(GAME_LOG_DIR, exist_ok=True)
        self.move_start_time = None

        print("=" * 70)
        print("GEDA五子棋AI v1.1 - 带防守评估")
        print("记忆池 | 怀疑语义化 | 动态位置评分 | 进攻+防守价值")
        print("=" * 70)
        print(f"黑方: {'人类' if black_player == PlayerType.HUMAN else 'GEDA AI'}")
        print(f"白方: {'人类' if white_player == PlayerType.HUMAN else 'GEDA AI'}")
        print("=" * 70)

    def draw_board(self):
        self.screen.fill(BOARD_BROWN)
        for i in range(BOARD_SIZE):
            x = MARGIN + i * GRID_SIZE
            pygame.draw.line(self.screen, BLACK, (x, MARGIN), (x, MARGIN + BOARD_HEIGHT), 2)
            y = MARGIN + i * GRID_SIZE
            pygame.draw.line(self.screen, BLACK, (MARGIN, y), (MARGIN + BOARD_WIDTH, y), 2)
        star_positions = [(7,7), (3,3), (11,3), (3,11), (11,11)]
        for (sx, sy) in star_positions:
            cx = MARGIN + sx * GRID_SIZE
            cy = MARGIN + sy * GRID_SIZE
            pygame.draw.circle(self.screen, BLACK, (cx, cy), 5)

    def draw_pieces(self):
        for piece in self.board.pieces:
            if piece.captured:
                continue
            x, y = piece.position
            sx = MARGIN + x * GRID_SIZE
            sy = MARGIN + y * GRID_SIZE
            center = (sx, sy)
            radius = GRID_SIZE // 2 - 4
            color = BLACK_PIECE_COLOR if piece.is_black else WHITE_PIECE_COLOR
            border = DARK_GRAY if piece.is_black else LIGHT_GRAY
            pygame.draw.circle(self.screen, color, center, radius)
            pygame.draw.circle(self.screen, border, center, radius, 2)

    def draw_last_move(self):
        if self.last_move and self.highlight_last_move:
            x, y = self.last_move
            sx = MARGIN + x * GRID_SIZE
            sy = MARGIN + y * GRID_SIZE
            pygame.draw.circle(self.screen, YELLOW, (sx, sy), GRID_SIZE//2, 3)

    def draw_position_scores(self):
        if not self.current_ai:
            return
        scores = self.current_ai.position_scores
        min_s, max_s = -30, 30
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board.get_piece_at(x, y) is not None:
                    continue
                val = scores[y, x]
                if val >= 0:
                    intensity = int(255 * val / max_s)
                    color = (255, 255-intensity, 255-intensity)
                else:
                    intensity = int(255 * (-val) / (-min_s))
                    color = (255-intensity, 255-intensity, 255)
                sx = MARGIN + x * GRID_SIZE - GRID_SIZE//2
                sy = MARGIN + y * GRID_SIZE - GRID_SIZE//2
                s = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
                s.fill((*color, 100))
                self.screen.blit(s, (sx, sy))

    def draw_info_panel(self):
        panel = pygame.Rect(INFO_PANEL_X, 10, INFO_PANEL_WIDTH, SCREEN_HEIGHT - 20)
        pygame.draw.rect(self.screen, DARK_GRAY, panel)
        pygame.draw.rect(self.screen, LIGHT_GRAY, panel, 2)
        y = 20

        title = self.large_font.render("GEDA五子棋监视器", True, CYAN)
        self.screen.blit(title, (INFO_PANEL_X + 20, y))
        y += 40

        turn = "黑方" if self.board.current_player else "白方"
        turn_color = BLACK_PIECE_COLOR if self.board.current_player else WHITE_PIECE_COLOR
        turn_text = self.font.render(f"当前回合: {turn}", True, turn_color)
        self.screen.blit(turn_text, (INFO_PANEL_X + 20, y))
        y += 30

        state_text = self.font.render(f"状态: {self.game_state.name}", True, WHITE)
        self.screen.blit(state_text, (INFO_PANEL_X + 20, y))
        y += 25
        move_text = self.font.render(f"步数: {self.move_count}", True, YELLOW)
        self.screen.blit(move_text, (INFO_PANEL_X + 20, y))
        y += 30

        black_cnt = sum(1 for p in self.board.pieces if p.is_black)
        white_cnt = len(self.board.pieces) - black_cnt
        piece_text = self.font.render(f"黑{black_cnt} 白{white_cnt}", True, WHITE)
        self.screen.blit(piece_text, (INFO_PANEL_X + 20, y))
        y += 40

        score = self.board.evaluate_board_detailed(True)
        score_text = self.small_font.render(f"黑方优势: {score:+.2f}", True, GREEN if score>0 else RED)
        self.screen.blit(score_text, (INFO_PANEL_X + 20, y))
        y += 25

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

                doubt_color = RED if info['doubt_triggered'] else GREEN
                doubt_text = self.small_font.render(f"怀疑: {'是' if info['doubt_triggered'] else '否'}", True, doubt_color)
                self.screen.blit(doubt_text, (INFO_PANEL_X + 30, y))
                y += 22
                if info['doubt_triggered']:
                    reason_text = self.tiny_font.render(f"原因: {info['doubt_reason']}", True, LIGHT_GRAY)
                    self.screen.blit(reason_text, (INFO_PANEL_X + 40, y))
                    y += 18

                if info['memory_hit']:
                    mem_text = self.small_font.render(f"记忆命中 (成功率 {info['memory_success_rate']:.0%})", True, LIGHT_BLUE)
                    self.screen.blit(mem_text, (INFO_PANEL_X + 30, y))
                    y += 22

                move_desc = info['move_description'] if info['move_description'] else "无"
                move_text = self.small_font.render(f"最终走法: {move_desc}", True, YELLOW)
                self.screen.blit(move_text, (INFO_PANEL_X + 30, y))
                y += 22

                # 显示进攻和防守得分
                attack = info.get('attack_delta', 0)
                defense = info.get('defense_delta', 0)
                total = info['delta']
                attack_color = GREEN if attack > 0 else RED
                defense_color = GREEN if defense > 0 else RED
                self.screen.blit(self.tiny_font.render(f"进攻: {attack:+.1f}  防守: {defense:+.1f}", True, WHITE), (INFO_PANEL_X + 30, y))
                y += 20
                delta_color = GREEN if total > 0 else RED
                delta_text = self.small_font.render(f"综合增值: {total:+.1f}", True, delta_color)
                self.screen.blit(delta_text, (INFO_PANEL_X + 30, y))
                y += 22

                rate = self.current_ai.success_count / max(self.current_ai.total_attempts, 1)
                rate_color = GREEN if rate > 0.6 else RED if rate < 0.3 else YELLOW
                rate_text = self.small_font.render(f"历史成功率: {rate:.1%}", True, rate_color)
                self.screen.blit(rate_text, (INFO_PANEL_X + 30, y))
                y += 22

                gene_len = len(self.current_ai.gene_chain)
                mem_cnt = len(self.current_ai.long_term_memory['game_patterns'])
                self.screen.blit(self.small_font.render(f"基因长度: {gene_len}", True, YELLOW), (INFO_PANEL_X + 30, y))
                y += 20
                self.screen.blit(self.small_font.render(f"记忆模式: {mem_cnt}", True, LIGHT_BLUE), (INFO_PANEL_X + 30, y))
                y += 20

                pool_size = len(self.current_ai.memory_gene_pool)
                self.screen.blit(self.small_font.render(f"记忆池片段: {pool_size}", True, ORANGE), (INFO_PANEL_X + 30, y))
                y += 20

                su = self.current_ai.doubt_strategy_utility
                util_text = f"怀疑策略:A:{su.get('aggressive',0.5):.2f} C:{su.get('conservative',0.5):.2f} T:{su.get('flexible',0.5):.2f} X:{su.get('explorative',0.5):.2f}"
                self.screen.blit(self.tiny_font.render(util_text, True, LIGHT_BLUE), (INFO_PANEL_X + 30, y))
                y += 18
                # 显示防守权重
                self.screen.blit(self.tiny_font.render(f"防守权重: {self.current_ai.defense_weight:.2f}", True, CYAN), (INFO_PANEL_X + 30, y))
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
            "鼠标点击落子",
            "R: 重开  空格: 暂停",
            "Z: 悔棋(人类)",
            "S: 保存记忆  L: 加载",
            "H: 高亮最后一步  P: 切换动态评分显示",
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
        if self.game_state == GameState.BLACK_WIN:
            msg = "黑方获胜！"
            color = BLACK_PIECE_COLOR
        elif self.game_state == GameState.WHITE_WIN:
            msg = "白方获胜！"
            color = WHITE_PIECE_COLOR
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

    def handle_click(self, pos):
        if self.game_state != GameState.PLAYING or self.game_paused:
            return
        current_is_ai = (self.board.current_player and self.black_player == PlayerType.GEDA_AI) or \
                        (not self.board.current_player and self.white_player == PlayerType.GEDA_AI)
        if current_is_ai:
            return
        x, y = pos
        grid_x = round((x - MARGIN) / GRID_SIZE)
        grid_y = round((y - MARGIN) / GRID_SIZE)
        if 0 <= grid_x < BOARD_SIZE and 0 <= grid_y < BOARD_SIZE:
            if self.board.get_piece_at(grid_x, grid_y) is None:
                self.execute_move(grid_x, grid_y)

    def execute_move(self, x, y):
        moving_player_is_black = self.board.current_player

        score_before = self.board.evaluate_board_detailed(moving_player_is_black)
        fen_before = self.board.extract_situation_signature()

        if not self.board.place_piece(x, y):
            return False

        score_after = self.board.evaluate_board_detailed(moving_player_is_black)
        fen_after = self.board.extract_situation_signature()
        delta = score_after - score_before

        is_win = self.board.game_state != GameState.PLAYING

        think_time = 0.0
        if self.move_start_time is not None:
            think_time = time.time() - self.move_start_time
            self.move_start_time = None

        move_record = {
            'move_number': self.move_count + 1,
            'player': '黑' if moving_player_is_black else '白',
            'position': (x, y),
            'fen_before': fen_before,
            'fen_after': fen_after,
            'score_before': round(score_before, 2),
            'score_after': round(score_after, 2),
            'delta': round(delta, 2),
            'is_win': is_win,
            'think_time': round(think_time, 2),
            'timestamp': time.time()
        }

        current_is_ai = (moving_player_is_black and self.black_player == PlayerType.GEDA_AI) or \
                        (not moving_player_is_black and self.white_player == PlayerType.GEDA_AI)
        if current_is_ai:
            ai = self.black_ai if moving_player_is_black else self.white_ai
            if ai:
                move_record['ai_decision'] = ai.last_decision_info.copy()
                dummy = ChessPiece(moving_player_is_black, (-1,-1))
                ai._learn_from_move(self.board, (dummy, (x,y)), delta)

        self.game_state = self.board.game_state
        self.last_move = (x, y)
        self.move_count += 1

        self.game_history.append(move_record)

        if self.game_state != GameState.PLAYING:
            self.on_game_end()

        return True

    def ai_make_move(self):
        if self.game_state != GameState.PLAYING or self.game_paused:
            return
        legal_moves = self.board.get_legal_moves(self.board.current_player)
        if not legal_moves:
            self.game_state = GameState.DRAW
            self.on_game_end()
            return
        self.current_ai = self.black_ai if self.board.current_player else self.white_ai
        if not self.current_ai:
            return
        self.ai_thinking = True
        self.ai_move_start_time = time.time()
        move = self.current_ai.make_decision(self.board)
        self.ai_thinking_time = time.time() - self.ai_move_start_time
        self.ai_thinking = False
        if move:
            dummy, target = move
            x, y = target
            self.execute_move(x, y)
            if random.random() < 0.2:
                self.current_ai.mutate_gene_chain()
            if random.random() < 0.1:
                self.current_ai._mutate_doubt_gene_chain()
        else:
            self.game_state = GameState.DRAW
            self.on_game_end()

    def undo_move(self):
        if self.move_count == 0 or self.game_state != GameState.PLAYING:
            return
        current_is_ai = (self.board.current_player and self.black_player == PlayerType.GEDA_AI) or \
                        (not self.board.current_player and self.white_player == PlayerType.GEDA_AI)
        if not current_is_ai:
            self.board.undo_move()
            self.move_count -= 1
            if self.game_history:
                self.game_history.pop()
            print("悔棋一步")
        self.game_state = self.board.game_state
        if self.board.move_history:
            last = self.board.move_history[-1]
            self.last_move = last.position
        else:
            self.last_move = None

    def restart_game(self):
        if self.black_ai and self.black_ai.game_moves:
            self.black_ai.apply_endgame_credit(False)
        if self.white_ai and self.white_ai.game_moves:
            self.white_ai.apply_endgame_credit(False)

        black_genes = self.black_ai.gene_chain.copy() if self.black_ai else None
        white_genes = self.white_ai.gene_chain.copy() if self.white_ai else None
        if self.black_ai:
            self.black_ai.save_long_term_memory()
        if self.white_ai:
            self.white_ai.save_long_term_memory()

        if self.game_history:
            self.save_game_record()

        self.board = ChessBoard()
        self.game_state = GameState.PLAYING
        self.selected_piece = None
        self.last_move = None
        self.move_count = 0
        self.game_history = []
        self.ai_thinking = False
        if self.black_player == PlayerType.GEDA_AI:
            self.black_ai = GEDA_GeneSystem(is_black=True)
            if black_genes:
                self.black_ai.gene_chain = black_genes
        if self.white_player == PlayerType.GEDA_AI:
            self.white_ai = GEDA_GeneSystem(is_black=False)
            if white_genes:
                self.white_ai.gene_chain = white_genes
        self.current_ai = self.black_ai if self.board.current_player else self.white_ai
        self.move_start_time = None
        print("游戏重新开始")

    def on_game_end(self):
        print(f"游戏结束！{self.game_state.name}")
        winner_is_black = (self.game_state == GameState.BLACK_WIN)
        if self.black_ai:
            self.black_ai.apply_endgame_credit(winner_is_black)
        if self.white_ai:
            self.white_ai.apply_endgame_credit(not winner_is_black)

        self.save_game_record()
        if self.black_ai:
            self.black_ai.save_long_term_memory()
        if self.white_ai:
            self.white_ai.save_long_term_memory()

    def save_game_record(self):
        if not self.game_history:
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        filename = f"game_{timestamp}.txt"
        filepath = os.path.join(GAME_LOG_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("                          GEDA 五子棋AI 对局记录\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"对局时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"黑方: {'人类' if self.black_player == PlayerType.HUMAN else 'GEDA AI'}\n")
            f.write(f"白方: {'人类' if self.white_player == PlayerType.HUMAN else 'GEDA AI'}\n")
            f.write(f"结果: {self.game_state.name}\n")
            f.write(f"总步数: {self.move_count}\n\n")

            f.write("【对局过程】\n")
            f.write("-" * 60 + "\n")
            f.write("步数 | 走棋方 | 位置  | 评分变化 | 思考时间\n")
            f.write("-" * 60 + "\n")
            for move in self.game_history:
                delta_str = f"{move['delta']:+7.2f}"
                time_str = f"{move['think_time']:5.2f}s" if move['think_time'] > 0 else "     "
                f.write(
                    f"{move['move_number']:3d}  |  {move['player']}    | ({move['position'][0]},{move['position'][1]}) "
                    f"|  {delta_str:>6}  | {time_str}\n")

            f.write("\n\n")
            f.write("=" * 80 + "\n")
            f.write("【每一步的完整监控数据】\n")
            f.write("=" * 80 + "\n\n")

            for move in self.game_history:
                f.write(
                    f"第{move['move_number']}步: {move['player']} 落子 ({move['position'][0]},{move['position'][1]})\n")
                f.write(f"  移动前评分: {move['score_before']:8.2f} (黑方视角)\n")
                f.write(f"  移动后评分: {move['score_after']:8.2f} (黑方视角)\n")
                f.write(f"  评分增量:   {move['delta']:+8.2f}\n")
                f.write(f"  局面特征码: {move['fen_before']} → {move['fen_after']}\n")
                f.write(f"  思考时间: {move['think_time']:.2f} 秒\n")

                if 'ai_decision' in move:
                    ai = move['ai_decision']
                    f.write("  【AI决策监控】\n")
                    f.write(f"    策略模式: {ai['strategy_mode']}\n")
                    f.write(f"    环境压力: {ai['environment_pressure']:.2f}\n")
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
                    f.write(f"    进攻得分: {ai.get('attack_delta',0):+.1f}\n")
                    f.write(f"    防守得分: {ai.get('defense_delta',0):+.1f}\n")
                    f.write(f"    综合增值: {ai['delta']:+.1f}\n")

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
                elif event.key == pygame.K_z:
                    if not self.ai_thinking:
                        self.undo_move()
                elif event.key == pygame.K_s:
                    if self.black_ai:
                        self.black_ai.save_long_term_memory()
                    if self.white_ai:
                        self.white_ai.save_long_term_memory()
                    print("手动保存AI记忆")
                elif event.key == pygame.K_l:
                    if self.black_ai:
                        self.black_ai.load_long_term_memory()
                    if self.white_ai:
                        self.white_ai.load_long_term_memory()
                    print("手动加载AI记忆")
                elif event.key == pygame.K_p:
                    self.show_position_scores = not self.show_position_scores
                    print(f"动态评分显示: {'开启' if self.show_position_scores else '关闭'}")

    def update(self):
        if self.game_paused or self.game_state != GameState.PLAYING:
            return
        legal_moves = self.board.get_legal_moves(self.board.current_player)
        if not legal_moves:
            self.game_state = GameState.DRAW
            self.on_game_end()
            return
        current_is_ai = (self.board.current_player and self.black_player == PlayerType.GEDA_AI) or \
                        (not self.board.current_player and self.white_player == PlayerType.GEDA_AI)
        if current_is_ai and not self.ai_thinking:
            if not hasattr(self, 'ai_delay') or time.time() - self.ai_delay > 0.5:
                self.ai_make_move()
                self.ai_delay = time.time()

    def draw(self):
        self.draw_board()
        if self.show_position_scores:
            self.draw_position_scores()
        self.draw_pieces()
        self.draw_last_move()
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
        if self.black_ai:
            self.black_ai.save_long_term_memory()
        if self.white_ai:
            self.white_ai.save_long_term_memory()
        print("游戏结束，再见！")


# ============================================
# 7. 主程序
# ============================================
def main():
    os.makedirs(MEMORY_DIR, exist_ok=True)
    game = GomokuGameGUI(
        black_player=PlayerType.HUMAN,
        white_player=PlayerType.GEDA_AI
    )
    game.run()


if __name__ == "__main__":
    main()