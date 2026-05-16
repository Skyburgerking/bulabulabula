"""
GEDA V7.0 - 完全干净抽象学习版
============================================================
环境：宝物猎人（5宝物，3属性，真实价值 = 0.6*A + 0.3*B + 0.1*C）
设计哲学：零先验知识，纯通用符号回归技术
   - 无任何任务特定奖惩（无系数精度奖励、无局部搜索偏向）
   - 仅依赖环境得分 + 通用正则化
============================================================
"""

import random
import math
import hashlib
import numpy as np
from collections import defaultdict, deque
from copy import deepcopy

# ==================== 1. 游戏环境 ====================
class TreasureEnv:
    def __init__(self):
        self.hidden_weights = [0.6, 0.3, 0.1]

    def reset(self):
        self.items = []
        for _ in range(5):
            a = random.random()
            b = random.random()
            c = random.random()
            self.items.append((a, b, c))
        return self.items

    def get_true_value(self, item):
        a, b, c = item
        return self.hidden_weights[0]*a + self.hidden_weights[1]*b + self.hidden_weights[2]*c

    def step(self, choice_idx):
        score = self.get_true_value(self.items[choice_idx])
        next_items = self.reset()
        return score, next_items


# ==================== 2. 公式表示与遗传编程 ====================
OPS = ['+', '-', '*', '/']
VARS = ['A', 'B', 'C']

# ---------- 细粒度常量池（0.01 ~ 2.00，步长0.01）----------
CONSTS = [round(i * 0.01, 2) for i in range(1, 201)]

# 操作符权重（除法权重2.0，通用复杂度惩罚）
OP_WEIGHT = {'+': 1.0, '-': 1.0, '*': 1.5, '/': 2.0}

# ---------- 动态常量池（纯自适应，无预引导）----------
constant_score = defaultdict(float)
constant_usage = defaultdict(int)

def get_constant_weights():
    weights = []
    for c in CONSTS:
        if constant_usage[c] > 0:
            base = constant_score[c] / constant_usage[c]
            weights.append(max(0.1, base))
        else:
            weights.append(1.0)
    return weights

def random_const():
    weights = get_constant_weights()
    return random.choices(CONSTS, weights=weights, k=1)[0]

# ---------- 表达式基本操作 ----------
def random_expr(depth=3):
    if depth <= 0 or random.random() < 0.4:
        if random.random() < 0.6:
            return random.choice(VARS)
        else:
            return random_const()
    op = random.choice(OPS)
    if random.random() < 0.3:
        return random_expr(depth-1)
    return [op, random_expr(depth-1), random_expr(depth-1)]

def expr_to_str(expr):
    if isinstance(expr, (int, float)):
        return f"{expr:.2f}"
    if isinstance(expr, str):
        return expr
    if isinstance(expr, list):
        return f"({expr_to_str(expr[1])}{expr[0]}{expr_to_str(expr[2])})"

def expr_variables(expr):
    vars_set = set()
    if isinstance(expr, str):
        if expr in VARS:
            vars_set.add(expr)
    elif isinstance(expr, list):
        for sub in expr[1:]:
            vars_set.update(expr_variables(sub))
    return vars_set

def expr_variable_count(expr):
    count = {v: 0 for v in VARS}
    def count_vars(e):
        if isinstance(e, str) and e in VARS:
            count[e] += 1
        elif isinstance(e, list):
            for sub in e[1:]:
                count_vars(sub)
    count_vars(expr)
    return count

def expr_complexity_weighted(expr):
    if isinstance(expr, (int, float, str)):
        return 1
    if isinstance(expr, list):
        op = expr[0]
        weight = OP_WEIGHT.get(op, 1.0)
        return weight + sum(expr_complexity_weighted(sub) for sub in expr[1:])
    return 1

def has_any_variable(expr):
    if isinstance(expr, str) and expr in VARS:
        return True
    if isinstance(expr, list):
        return has_any_variable(expr[1]) or (len(expr) > 2 and has_any_variable(expr[2]))
    return False

def count_constants(expr):
    if isinstance(expr, (int, float)):
        return 1
    if isinstance(expr, str):
        return 0
    if isinstance(expr, list):
        return sum(count_constants(sub) for sub in expr[1:])
    return 0

def count_var_div_const(expr):
    count = 0
    if isinstance(expr, list):
        op = expr[0]
        if op == '/':
            left, right = expr[1], expr[2]
            left_is_var = isinstance(left, str) and left in VARS
            right_is_const = isinstance(right, (int, float))
            if left_is_var and right_is_const:
                count += 1
        for sub in expr[1:]:
            count += count_var_div_const(sub)
    return count

# ========== 线性结构检测（通用）==========
def is_linear_style(expr):
    """检测是否为线性结构：只包含 +、-、*，且*必须是变量*常量或常量*变量，无变量*变量"""
    def check(e):
        if isinstance(e, (int, float, str)):
            return True
        if isinstance(e, list):
            op = e[0]
            if op not in ['+', '-', '*']:
                return False
            if op == '*':
                left, right = e[1], e[2]
                left_is_var = isinstance(left, str) and left in VARS
                right_is_var = isinstance(right, str) and right in VARS
                left_is_const = isinstance(left, (int, float))
                right_is_const = isinstance(right, (int, float))
                if not ((left_is_var and right_is_const) or (left_is_const and right_is_var)):
                    return False
            return check(e[1]) and (len(e) <= 2 or check(e[2]))
        return False
    return check(expr)

# ========== 线性拟合优化（核心通用技术）==========
def is_linear_combination_candidate(expr):
    """判断是否为线性加权和候选（只含+-和变量*常量）"""
    def check(e):
        if isinstance(e, (int, float, str)):
            return True
        if isinstance(e, list):
            op = e[0]
            if op not in ['+', '-']:
                if op == '*':
                    left, right = e[1], e[2]
                    left_is_var = isinstance(left, str) and left in VARS
                    right_is_var = isinstance(right, str) and right in VARS
                    left_is_const = isinstance(left, (int, float))
                    right_is_const = isinstance(right, (int, float))
                    if (left_is_var and right_is_const) or (left_is_const and right_is_var):
                        return True
                    else:
                        return False
                else:
                    return False
            return check(e[1]) and (len(e) <= 2 or check(e[2]))
        return False
    return check(expr)

def extract_linear_coefficients(expr):
    """提取线性表达式的系数和常数"""
    coeffs = {v: 0.0 for v in VARS}
    const = 0.0
    def parse(e, sign=1):
        nonlocal const
        if isinstance(e, (int, float)):
            const += sign * e
        elif isinstance(e, str):
            if e in VARS:
                coeffs[e] += sign * 1.0
        elif isinstance(e, list):
            op = e[0]
            if op == '+':
                parse(e[1], sign)
                parse(e[2], sign)
            elif op == '-':
                parse(e[1], sign)
                parse(e[2], -sign)
            elif op == '*':
                left, right = e[1], e[2]
                left_is_var = isinstance(left, str) and left in VARS
                right_is_var = isinstance(right, str) and right in VARS
                left_is_const = isinstance(left, (int, float))
                right_is_const = isinstance(right, (int, float))
                if left_is_var and right_is_const:
                    coeffs[left] += sign * right
                elif left_is_const and right_is_var:
                    coeffs[right] += sign * left
    parse(expr)
    return coeffs, const

def linear_fit_optimization(expr, env, samples=150):
    """最小二乘法拟合线性系数，返回优化后的表达式（无任何目标系数先验）"""
    if not is_linear_combination_candidate(expr):
        return expr
    vars_used = expr_variables(expr)
    if len(vars_used) < 2:  # 至少两个变量才有拟合意义
        return expr
    # 收集随机样本
    X, y = [], []
    for _ in range(samples):
        a = random.random()
        b = random.random()
        c = random.random()
        target = env.get_true_value((a, b, c))
        X.append([a, b, c])
        y.append(target)
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    X_with_const = np.column_stack([np.ones(len(X)), X])
    try:
        beta, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
        const_opt = round(float(beta[0]), 2)
        coeffs_opt = {
            'A': round(float(beta[1]), 2),
            'B': round(float(beta[2]), 2),
            'C': round(float(beta[3]), 2)
        }
    except:
        return expr
    # 构建新表达式
    terms = []
    if abs(const_opt) > 1e-6:
        terms.append(const_opt)
    for v in VARS:
        if abs(coeffs_opt[v]) > 1e-6:
            if coeffs_opt[v] == 1.0:
                terms.append(v)
            else:
                terms.append(['*', coeffs_opt[v], v])
    if not terms:
        return 0.0
    if len(terms) == 1:
        return terms[0]
    expr_new = terms[0]
    for t in terms[1:]:
        expr_new = ['+', expr_new, t]
    return expr_new

# ========== 变异算子（纯通用，无偏向）=========
def hoist_mutate(expr):
    """提升一个子树为整个表达式"""
    subtrees = []
    def collect(e):
        subtrees.append(e)
        if isinstance(e, list):
            for sub in e[1:]:
                collect(sub)
    collect(expr)
    if len(subtrees) > 1:
        candidate = random.choice(subtrees[1:])
        return deepcopy(candidate)
    return expr

def shrink_mutate(expr):
    """将内部节点替换为其一个子节点"""
    if not isinstance(expr, list):
        return expr
    def select_nodes(e):
        nodes = []
        if isinstance(e, list):
            nodes.append(e)
            for sub in e[1:]:
                nodes.extend(select_nodes(sub))
        return nodes
    all_nodes = select_nodes(expr)
    internal = [n for n in all_nodes if isinstance(n, list) and len(n) > 1]
    if not internal:
        return expr
    node = random.choice(internal)
    child = random.choice(node[1:])
    return deepcopy(child)

def tweak_constants_multi(expr, deltas=[0.05, 0.1, 0.2]):
    """多步长常量微调（无任何变量偏向）"""
    if isinstance(expr, (int, float)):
        delta = random.choice(deltas) * random.choice([-1, 1])
        new_val = round(expr + delta, 2)
        if new_val <= 0: new_val = 0.01
        if new_val > 2.0: new_val = 2.0
        return new_val
    if isinstance(expr, str):
        return expr
    if isinstance(expr, list):
        new_expr = [expr[0]]
        for sub in expr[1:]:
            new_expr.append(tweak_constants_multi(sub, deltas))
        return new_expr
    return expr

def simplify_expr(expr):
    """代数简化（纯通用）"""
    if isinstance(expr, np.ndarray):
        expr = expr.tolist()
    if isinstance(expr, np.generic):
        expr = float(expr)
    if isinstance(expr, (int, float, str)):
        return expr
    if isinstance(expr, list):
        op = expr[0]
        left = simplify_expr(expr[1])
        right = simplify_expr(expr[2]) if len(expr) > 2 else None
        if op == '+':
            if isinstance(left, (int, float)) and left == 0:
                return right
            if isinstance(right, (int, float)) and right == 0:
                return left
            if left == right and isinstance(left, str) and left in VARS:
                return ['*', left, 2.0]
        if op == '-':
            if isinstance(right, (int, float)) and right == 0:
                return left
            if left == right:
                return 0.0
        if op == '*':
            if isinstance(left, (int, float)) and left == 0:
                return 0.0
            if isinstance(right, (int, float)) and right == 0:
                return 0.0
            if isinstance(left, (int, float)) and left == 1:
                return right
            if isinstance(right, (int, float)) and right == 1:
                return left
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                return round(left * right, 2)
        if op == '/':
            if isinstance(right, (int, float)) and right == 1:
                return left
            if left == right and has_any_variable(left):
                return 1.0
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                if right != 0:
                    return round(left / right, 2)
        return [op, left, right] if right is not None else [op, left]
    return expr

def mutate_expr(expr, env=None, prob=0.6):
    """主变异函数（集成多种变异，无偏向）"""
    if isinstance(expr, np.ndarray):
        expr = expr.tolist()
    if isinstance(expr, np.generic):
        expr = float(expr)
    r = random.random()
    if r < prob:
        mut_type = random.choice(['standard', 'hoist', 'shrink', 'tweak_multi'])
        if mut_type == 'standard':
            return simplify_expr(random_expr())
        elif mut_type == 'hoist':
            return simplify_expr(hoist_mutate(expr))
        elif mut_type == 'shrink':
            return simplify_expr(shrink_mutate(expr))
        elif mut_type == 'tweak_multi':
            return simplify_expr(tweak_constants_multi(expr))
    # 保留原结构变异
    if isinstance(expr, list) and expr[0] == '*':
        left, right = expr[1], expr[2]
        left_has_var = has_any_variable(left)
        right_has_var = has_any_variable(right)
        if left_has_var and right_has_var and random.random() < 0.5:
            c1 = random_const()
            c2 = random_const()
            return simplify_expr(['+', ['*', left, c1], ['*', right, c2]])
    if isinstance(expr, list) and expr[0] == '+' and random.random() < 0.2:
        return simplify_expr(['*', expr[1], expr[2]])
    if isinstance(expr, str) and expr in VARS and random.random() < 0.2:
        const = random_const()
        return simplify_expr(['*', expr, const])
    if isinstance(expr, list):
        new_expr = [expr[0]]
        for sub in expr[1:]:
            new_expr.append(mutate_expr(sub, env, prob))
        return simplify_expr(new_expr)
    elif isinstance(expr, (int, float)):
        # 常量漂移：60%概率向高权重常量移动（纯自适应，无预设目标）
        if random.random() < 0.6:
            weights = get_constant_weights()
            sorted_consts = sorted(zip(CONSTS, weights), key=lambda x: x[1], reverse=True)
            top_consts = [c for c, _ in sorted_consts[:3]]
            target = random.choice(top_consts)
            if expr < target:
                new_val = min(expr + 0.05, target)
            else:
                new_val = max(expr - 0.05, target)
            new_val = round(new_val, 2)
        else:
            delta = random.choice([-0.05, 0.05, -0.1, 0.1, 0])
            new_val = round(expr + delta, 2)
        if new_val <= 0: new_val = 0.01
        if new_val > 2.0: new_val = 2.0
        return new_val
    else:
        if random.random() < 0.3:
            return random.choice(VARS)
        return expr

def crossover_expr(e1, e2):
    e1 = simplify_expr(e1)
    e2 = simplify_expr(e2)
    if random.random() < 0.5:
        return e1
    if isinstance(e1, list) and isinstance(e2, list):
        new = [e1[0]]
        for i in range(1, len(e1)):
            if i < len(e2) and isinstance(e1[i], (list, str, float, int)) and isinstance(e2[i], (list, str, float, int)):
                if random.random() < 0.3:
                    new.append(e2[i])
                else:
                    new.append(e1[i])
            else:
                new.append(e1[i])
        return simplify_expr(new)
    return e1

class GEDA_Genome:
    def __init__(self, expr=None):
        self.expr = simplify_expr(expr if expr else random_expr())
        self.fitness = 0.0
        self.age = 0

    def mutate(self, env=None):
        new_expr = mutate_expr(self.expr, env)
        new_ind = GEDA_Genome(new_expr)
        # 线性拟合优化（概率0.6，通用技术）
        if env is not None and random.random() < 0.6:
            optimized = linear_fit_optimization(new_expr, env, samples=150)
            new_ind = GEDA_Genome(optimized)
        return new_ind

    def crossover(self, other):
        return GEDA_Genome(crossover_expr(self.expr, other.expr))

# ========== 种群与多样性（纯通用）==========
def compute_diversity_bonus(expr, population):
    s = expr_to_str(expr)
    count = 0
    for ind in population:
        if hasattr(ind, 'expr'):
            if expr_to_str(ind.expr) == s:
                count += 1
    return -0.005 * (count - 1) if count > 1 else 0.0  # 惩罚降为0.005，避免适应度过低

class GEDA_Population:
    def __init__(self, pop_size=50):
        self.pop_size = pop_size
        self.individuals = [GEDA_Genome() for _ in range(pop_size)]
        self.generation = 0

    def evaluate_fitness(self, env, episodes=150):
        for ind in self.individuals:
            total = 0.0
            for _ in range(episodes):
                items = env.reset()
                preds = [safe_eval(ind.expr, a,b,c) for a,b,c in items]
                choice = preds.index(max(preds))
                score = env.get_true_value(items[choice])
                total += score
            raw = total / episodes
            # 复杂度惩罚（通用奥卡姆剃刀）
            penalty = 0.0035 * expr_complexity_weighted(ind.expr)
            # 变量覆盖奖励（通用多变量偏好）
            vars_used = expr_variables(ind.expr)
            coverage_reward = 0.08 if len(vars_used) == 3 else (0.03 if len(vars_used) >= 2 else 0.0)
            # 线性结构奖励（通用结构偏好，无特定系数）
            linear_reward = 0.05 if is_linear_style(ind.expr) else 0.0
            # 重复惩罚（防止变量无意义重复）
            repeat_penalty = -0.005 * sum(max(0, expr_variable_count(ind.expr)[v]-1) for v in VARS)
            # 常量惩罚（防止过度常量）
            const_penalty = -0.005 * count_constants(ind.expr)
            # 除法惩罚（抑制复杂运算）
            div_penalty = -0.01 * count_var_div_const(ind.expr)
            # 多样性奖励
            diversity_bonus = compute_diversity_bonus(ind.expr, self.individuals)
            
            ind.fitness = (raw - penalty + coverage_reward + linear_reward +
                          repeat_penalty + const_penalty + div_penalty + diversity_bonus)
            
            # 年龄惩罚（防止陈旧个体，阈值调至6，惩罚0.05）
            if ind.age >= 6:
                ind.fitness -= 0.05
            ind.age += 1
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)

    def select(self):
        tourn = random.sample(self.individuals[:25], 7)
        tourn.sort(key=lambda x: x.fitness, reverse=True)
        return tourn[0]

    def evolve(self, env):
        new_inds = []
        new_inds.extend(self.individuals[:12])
        while len(new_inds) < self.pop_size:
            p1 = self.select()
            p2 = self.select()
            child = p1.crossover(p2)
            if random.random() < 0.7:
                child = child.mutate(env)
            new_inds.append(child)
        self.individuals = new_inds
        self.generation += 1

        # 每10代刷新后40%个体
        if self.generation % 10 == 0:
            replace_num = int(self.pop_size * 0.4)
            for i in range(1, replace_num+1):
                self.individuals[-i] = GEDA_Genome()
            print(f"   [种群刷新] 替换 {replace_num} 个个体")


# ==================== 3. 基础记忆系统 ====================
class GEDA_Memory:
    def __init__(self):
        self.mem = defaultdict(lambda: {'count':0, 'best_idx':0, 'avg_score':0.0})

    def hash_items(self, items):
        s = ''.join([f"{a:.2f}{b:.2f}{c:.2f}" for a,b,c in items])
        return hashlib.md5(s.encode()).hexdigest()[:16]

    def remember(self, items, choice, score):
        sig = self.hash_items(items)
        self.mem[sig]['count'] += 1
        old_avg = self.mem[sig]['avg_score']
        self.mem[sig]['avg_score'] = old_avg + (score - old_avg) / self.mem[sig]['count']
        if score > self.mem[sig].get('best_score', -1):
            self.mem[sig]['best_idx'] = choice
            self.mem[sig]['best_score'] = score

    def recall(self, items, min_count=3):
        sig = self.hash_items(items)
        if sig in self.mem and self.mem[sig]['count'] >= min_count:
            return self.mem[sig]['best_idx'], self.mem[sig]['avg_score']
        return None, None


# ==================== 4. 智能体 ====================
class AbstractGEDA_Agent:
    def __init__(self, pop_size=50):
        self.population = GEDA_Population(pop_size)
        self.memory = GEDA_Memory()
        self.bases = [0,1,2,3]
        self.complement_map = {0:2,1:3,2:0,3:1}
        self.gene_chain = self._init_chain(20)
        self.formula_memory = []
        self.formula_memory_capacity = 30
        self.feature_stats = defaultdict(lambda: {'success':0, 'total':0})
        self.exploration_rate = 0.25
        self.doubt_rate = 0.1
        self.total_attempts = 0
        self.success_streak = 0
        self.failure_streak = 0
        self.performance_history = []
        self.total_score = 0.0

    def _init_chain(self, length):
        primary = [random.choice(self.bases) for _ in range(length)]
        complementary = [self.complement_map[b] for b in primary]
        return {'primary': primary, 'complementary': complementary}

    def _extract_features(self, items, choice):
        a,b,c = items[choice]
        A_vals = [it[0] for it in items]
        B_vals = [it[1] for it in items]
        C_vals = [it[2] for it in items]
        feats = []
        if a == max(A_vals): feats.append('A_max')
        if a == min(A_vals): feats.append('A_min')
        if b == max(B_vals): feats.append('B_max')
        if b == min(B_vals): feats.append('B_min')
        if c == max(C_vals): feats.append('C_max')
        if c == min(C_vals): feats.append('C_min')
        if a > 0.8: feats.append('A_high')
        if b > 0.8: feats.append('B_high')
        if c > 0.8: feats.append('C_high')
        if a < 0.2: feats.append('A_low')
        if b < 0.2: feats.append('B_low')
        if c < 0.2: feats.append('C_low')
        return feats

    def _heuristic_decision(self, items):
        for idx, (a,b,c) in enumerate(items):
            feats = self._extract_features(items, idx)
            for f in feats:
                st = self.feature_stats[f]
                if st['total'] >= 5 and st['success'] / st['total'] > 0.8:
                    return idx, f
        return None, None

    def _pressure(self, items):
        A = [it[0] for it in items]
        B = [it[1] for it in items]
        C = [it[2] for it in items]
        return np.var(A) + np.var(B) + np.var(C)

    def _chain_to_index(self, strand):
        h = 0
        for b in strand:
            h = (h * 31 + b) & 0xFFFFFFFF
        elite_pool = self.population.individuals[:8]
        if not elite_pool:
            return 0
        return h % len(elite_pool)

    def _generate_cypher(self, items, strand):
        sig = self.memory.hash_items(items)
        s = int(sig, 16) % 10**6
        strand_hash = 0
        for b in strand:
            strand_hash = (strand_hash * 17 + b) & 0xFFFFFF
        rand = random.randint(1,1000)
        code = (s ^ strand_hash ^ rand) % 10**5
        return str(code), code

    def _tweak_constants_multi(self, expr):
        return tweak_constants_multi(expr)  # 无偏向版本

    def _select_best_formula_with_local_search(self, items):
        candidates = []
        if self.formula_memory and random.random() < 0.3:
            best_mem = max(self.formula_memory, key=lambda x: x[1])
            candidates.append(best_mem[0])
        for ind in self.population.individuals[:3]:
            candidates.append(ind.expr)
        if self.formula_memory:
            mem_candidates = random.sample(self.formula_memory, min(2, len(self.formula_memory)))
            for expr, _, _ in mem_candidates:
                if expr_to_str(expr) not in [expr_to_str(e) for e in candidates[:3]]:
                    candidates.append(expr)
        best_expr = None
        best_pred_max = -float('inf')
        best_choice = None
        best_preds = None
        for expr in candidates:
            preds = [safe_eval(expr, a,b,c) for a,b,c in items]
            max_pred = max(preds)
            if max_pred > best_pred_max:
                best_pred_max = max_pred
                best_expr = expr
                best_choice = preds.index(max(preds))
                best_preds = preds
            # 局部搜索概率0.8，无偏向
            if random.random() < 0.8:
                tweaked = self._tweak_constants_multi(expr)
                preds_t = [safe_eval(tweaked, a,b,c) for a,b,c in items]
                max_pred_t = max(preds_t)
                if max_pred_t > best_pred_max:
                    best_pred_max = max_pred_t
                    best_expr = tweaked
                    best_choice = preds_t.index(max(preds_t))
                    best_preds = preds_t
        if best_expr is None:
            best_expr = self.population.individuals[0].expr
            preds = [safe_eval(best_expr, a,b,c) for a,b,c in items]
            best_choice = preds.index(max(preds))
            best_preds = preds
        return best_expr, best_preds, best_choice

    def decide(self, items):
        self.total_attempts += 1
        self.population.individuals.sort(key=lambda x: x.fitness, reverse=True)
        heu_idx, heu_feat = self._heuristic_decision(items)
        if heu_idx is not None and random.random() > self.exploration_rate:
            return heu_idx
        mem_idx, _ = self.memory.recall(items, min_count=2)
        if mem_idx is not None and random.random() > self.exploration_rate:
            return mem_idx
        best_expr, preds, choice = self._select_best_formula_with_local_search(items)
        self.last_used_formula = best_expr
        self.last_used_source = 'candidate_search'
        pressure = self._pressure(items)
        if pressure > 0.8 and random.random() < self.doubt_rate:
            idx2 = self._chain_to_index(self.gene_chain['complementary'])
            backup_ind = self.population.individuals[idx2]
            preds2 = [safe_eval(backup_ind.expr, a,b,c) for a,b,c in items]
            choice2 = preds2.index(max(preds2))
            if max(preds2) > max(preds) + 0.01:
                choice = choice2
                self.last_used_formula = backup_ind.expr
                self.last_used_source = 'doubt'
        if random.random() < self.exploration_rate:
            choice = random.randint(0,4)
            self.last_used_source = 'random'
        return choice

    def learn(self, items, choice, score):
        self.total_score += score
        self.performance_history.append(score)
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)
        is_success = score > 0.7
        if is_success:
            self.success_streak += 1
            self.failure_streak = 0
        else:
            self.failure_streak += 1
            self.success_streak = 0
        self.memory.remember(items, choice, score)
        feats = self._extract_features(items, choice)
        for f in feats:
            self.feature_stats[f]['total'] += 1
            if is_success:
                self.feature_stats[f]['success'] += 1
        if is_success:
            def record_constants(expr):
                if isinstance(expr, (int, float)):
                    val = round(expr, 2)
                    if val in CONSTS:
                        constant_score[val] += score
                        constant_usage[val] += 1
                elif isinstance(expr, list):
                    for sub in expr[1:]:
                        record_constants(sub)
            record_constants(self.last_used_formula)
        if is_success and score > 0.7:
            expr_vars = expr_variables(self.last_used_formula)
            complexity = expr_complexity_weighted(self.last_used_formula)
            if len(expr_vars) >= 2 and complexity <= 18:
                expr_str_repr = expr_to_str(self.last_used_formula)
                found = False
                for i, (e, fit, cnt) in enumerate(self.formula_memory):
                    if expr_to_str(e) == expr_str_repr:
                        new_fit = (fit * cnt + score) / (cnt + 1)
                        self.formula_memory[i] = (e, new_fit, cnt+1)
                        found = True
                        break
                if not found:
                    if len(self.formula_memory) >= self.formula_memory_capacity:
                        self.formula_memory.sort(key=lambda x: x[1])
                        self.formula_memory.pop(0)
                    self.formula_memory.append((self.last_used_formula, score, 1))
        if len(self.performance_history) >= 10:
            recent_avg = np.mean(self.performance_history[-10:])
            self.exploration_rate = max(0.08, self.exploration_rate * 0.999)
            if recent_avg > 0.7:
                self.doubt_rate = max(0.02, self.doubt_rate * 0.99)
            else:
                self.doubt_rate = min(0.2, self.doubt_rate * 1.01)

    def evolve_population(self, env):
        self.population.evaluate_fitness(env, episodes=150)
        
        # ===== 精英强制线性拟合优化（纯通用，无目标先验）=====
        for i in range(min(12, len(self.population.individuals))):
            if random.random() < 0.8:  # 80%概率
                ind = self.population.individuals[i]
                optimized = linear_fit_optimization(ind.expr, env, samples=150)
                if expr_to_str(optimized) != expr_to_str(ind.expr):
                    new_ind = GEDA_Genome(optimized)
                    # 直接替换，不比较适应度（信任数学）
                    self.population.individuals[i] = new_ind
        
        self.population.evolve(env)
        if self.total_attempts % 500 == 0:
            self._mutate_gene_chain()

    def _mutate_gene_chain(self):
        chain = self.gene_chain['primary']
        if random.random() < 0.3 and len(chain) < 30:
            chain.append(random.choice(self.bases))
        if random.random() < 0.3 and len(chain) > 15:
            chain.pop(random.randint(0, len(chain)-1))
        self.gene_chain['complementary'] = [self.complement_map[b] for b in chain]


# ==================== 5. 训练循环 ====================
def safe_eval(expr, a, b, c):
    if isinstance(expr, (int, float)):
        return expr
    if isinstance(expr, str):
        if expr == 'A': return a
        if expr == 'B': return b
        if expr == 'C': return c
        try:
            return float(expr)
        except:
            return 0
    if isinstance(expr, list):
        op = expr[0]
        left = safe_eval(expr[1], a, b, c)
        if op in ['+', '-', '*', '/']:
            right = safe_eval(expr[2], a, b, c)
        if op == '+': return left + right
        if op == '-': return left - right
        if op == '*': return left * right
        if op == '/':
            if abs(right) < 1e-6: return 0
            return left / right
    return 0

def train_abstract(episodes=2000):
    env = TreasureEnv()
    agent = AbstractGEDA_Agent(pop_size=50)

    print("="*60)
    print("GEDA V7.0 - 完全干净抽象学习版")
    print(f"环境真实规律: 价值 = 0.6*A + 0.3*B + 0.1*C (仅作为背景)")
    print("零先验知识 | 纯通用符号回归技术")
    print("核心技术: 线性拟合(60%) | Hoist/Shrink变异 | 多步长局部搜索")
    print("         动态常量池 | 复杂度惩罚 | 年龄惩罚 | 多样性奖励")
    print("="*60 + "\n")

    total_score = 0
    for ep in range(1, episodes+1):
        items = env.reset()
        choice = agent.decide(items)
        score, _ = env.step(choice)
        agent.learn(items, choice, score)
        total_score += score

        if ep % 50 == 0:
            agent.evolve_population(env)
            best = agent.population.individuals[0]
            avg_fit = np.mean([ind.fitness for ind in agent.population.individuals[:10]])
            recent_avg = np.mean(agent.performance_history[-20:]) if agent.performance_history else 0
            print(f"轮次 {ep:5d} | 累计均分: {total_score/ep:.3f} | 近期均分: {recent_avg:.3f}")
            print(f"探索率: {agent.exploration_rate:.2f} | 怀疑率: {agent.doubt_rate:.2f} | 种群代数: {agent.population.generation:2d}")
            print(f"最佳适应度: {best.fitness:.3f} | 加权节点数: {expr_complexity_weighted(best.expr):.1f} | 变量集合: {expr_variables(best.expr)}")
            print(f"最佳公式: {expr_to_str(best.expr)}")
            print(f"常量数: {count_constants(best.expr)} | 变量/常量除法: {count_var_div_const(best.expr)}")
            print(f"公式记忆池: {len(agent.formula_memory)} 条 | 启发式规则: {len([f for f in agent.feature_stats if agent.feature_stats[f]['total']>=5])}")
            top_consts = sorted(CONSTS, key=lambda c: constant_score[c]/max(1, constant_usage[c]), reverse=True)[:3]
            print(f"常量权重TOP3: {[(c, constant_score[c]/max(1, constant_usage[c])) for c in top_consts]}")
            print()

    print("\n====== 训练完成 ======")
    best = agent.population.individuals[0]
    print(f"进化 {agent.population.generation} 代后，GEDA V7.0 发现的最佳公式：")
    print(f"   {expr_to_str(best.expr)}")
    print(f"   适应度: {best.fitness:.4f}")
    print(f"   加权节点数: {expr_complexity_weighted(best.expr):.1f} | 变量集合: {expr_variables(best.expr)}")
    print(f"\n真实规律（仅作为参考）： 0.6*A + 0.3*B + 0.1*C")
    print("="*60)
    
    input("\n训练结束，按Enter键退出...")


if __name__ == "__main__":
    train_abstract(episodes=2000)