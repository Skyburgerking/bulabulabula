
import random
import itertools
from collections import defaultdict, Counter
import numpy as np
import hashlib
import math  # 需要导入math模块

class OptimizedMacroReinforcedGEDA:
    """
    优化版宏观正反馈与记忆基因链GEDA代理
    重点修复怀疑机制：用怀疑基因链反着推理
    """
    def __init__(self):
        # 1. 基础基因系统
        self.bases = ['A', 'G', 'C', 'T', 'X']
        self.complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'X': 'X'}
        self.gene_chain = self._initialize_gene_chain()
        self.doubt_gene_chain = self._initialize_doubt_gene_chain()  # 专门用于怀疑的基因链

        # 1.1 改进版记忆基因链
        self.memory_gene_pool = []
        self.memory_capacity = 30
        
        # 1.2 宏观正反馈系统
        self.success_streak = 0
        self.failure_streak = 0
        self.performance_history = []
        self.history_window = 20
        
        # 1.3 策略选择器
        self.strategy_mode = "balanced"
        self.mode_switch_threshold = 0.7

        # 2. 记忆库
        self.memory_dbs = {
            'success': defaultdict(list),
            'failure': defaultdict(list)
        }
        self.success_count = 0
        self.total_attempts = 0
        self.recent_success_rate = 0.0
        
        # 3. 增强的元认知模型
        self.pressure_correlations = Counter()
        self.rule_patterns = defaultdict(lambda: {'success': 0, 'failure': 0, 'total': 0})
        self.gene_segment_utility = defaultdict(float)
        self.current_active_strand_history = []

        # 4. 新增：怀疑状态追踪
        self.doubt_states = []  # 记录每次怀疑的状态
        self.doubt_effectiveness = 0.5  # 怀疑有效性（初始50%）

    def _initialize_gene_chain(self, length=20):
        """初始化主基因链"""
        primary_chain = [random.choice(self.bases) for _ in range(length)]
        complementary_chain = [self.complement_map[base] for base in primary_chain]
        return {'primary': primary_chain, 'complementary': complementary_chain}

    def _initialize_doubt_gene_chain(self):
        """初始化怀疑基因链 - 专门用于反向推理"""
        # 怀疑基因链的特点是：反逻辑、反直觉
        doubt_bases = []
        for _ in range(15):
            # 怀疑链更倾向于X（探索）和T（灵活），较少A（激进）
            weights = {'A': 0.1, 'G': 0.2, 'C': 0.2, 'T': 0.3, 'X': 0.2}
            base = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
            doubt_bases.append(base)
        
        return {'primary': doubt_bases, 'complementary': [self.complement_map[b] for b in doubt_bases]}

    def _calculate_pressure(self, game_state):
        """压力计算"""
        numbers = game_state['numbers']
        target = game_state['target']
        
        base_pressure = len(numbers) + sum([abs(n) for n in numbers if n != 0])
        
        min_possible = min(numbers)
        max_possible = max(numbers) * (len(numbers) - 1) if len(numbers) > 1 else max(numbers)
        
        if target < min_possible or target > max_possible * 2:
            base_pressure += 3
            
        unique_numbers = len(set(numbers))
        if unique_numbers / len(numbers) < 0.5:
            base_pressure += 2
            
        return base_pressure

    def _get_active_strand(self, pressure, game_state_key):
        """获取活性链"""
        self.current_active_strand_history = []
        
        if self.strategy_mode == "exploitative" and self.memory_gene_pool:
            best_memory = max(self.memory_gene_pool, key=lambda x: x[1])
            memory_segment = best_memory[0]
            print(f" [开发模式] 使用最优记忆片段 (效用值: {best_memory[1]:.2f})")
            chosen_chain = self._build_chain_from_memory(memory_segment, pressure)
            
        elif self.strategy_mode == "explorative" or random.random() < 0.3:
            if pressure > 5:
                chosen_chain = self.gene_chain['complementary']
            else:
                chosen_chain = self.gene_chain['primary'] if random.random() < 0.5 else self.gene_chain['complementary']
                    
            if self.strategy_mode == "explorative":
                print(f" [探索模式] 尝试新的基因组合")
                
        else:
            if game_state_key in self.memory_dbs['success'] and random.random() < 0.6:
                print(f" [记忆复用] 使用状态 {game_state_key} 的成功策略")
                chosen_chain = self.gene_chain['primary'] if random.random() < 0.5 else self.gene_chain['complementary']
            elif self.memory_gene_pool and random.random() < 0.4:
                memory_idx = random.randint(0, min(5, len(self.memory_gene_pool)-1))
                memory_segment = self.memory_gene_pool[memory_idx][0]
                chosen_chain = self._build_chain_from_memory(memory_segment, pressure)
                print(f" [记忆探索] 使用记忆库中的随机片段")
            else:
                chosen_chain = self.gene_chain['primary'] if random.random() < 0.5 else self.gene_chain['complementary']
        
        self.current_active_strand_history.append(chosen_chain)
        return chosen_chain

    def _build_chain_from_memory(self, memory_segment, pressure):
        """从记忆片段构建完整链"""
        if len(memory_segment) >= len(self.gene_chain['primary']):
            return list(memory_segment)[:len(self.gene_chain['primary'])]
        else:
            remaining_len = len(self.gene_chain['primary']) - len(memory_segment)
            if pressure > 7:
                fill_bases = [random.choice(['A', 'T']) for _ in range(remaining_len)]
            else:
                fill_bases = [random.choice(self.bases) for _ in range(remaining_len)]
            return list(memory_segment) + fill_bases

    def _validate_expression(self, expression_str, target_num, available_nums):
        """验证表达式"""
        try:
            allowed_chars = set('0123456789+-*/(). ')
            if not all(c in allowed_chars for c in expression_str.strip()):
                return False
            result = eval(expression_str)
            import re
            used_nums_in_expr = []
            for match in re.finditer(r'\d+\.?\d*', expression_str):
                num_str = match.group()
                used_nums_in_expr.append(float(num_str))
            if sorted(used_nums_in_expr) != sorted(available_nums):
                return False
            return abs(result - target_num) < 1e-9
        except (ValueError, SyntaxError, ZeroDivisionError):
            return False

    def _analyze_game_state_for_pressure(self, game_state, actual_pressure):
        """状态分析"""
        numbers = game_state['numbers']
        target = game_state['target']
        
        features = [
            f"num_count_{len(numbers)}",
            f"target_range_{'high' if target > 30 else 'low'}",
            f"sum_numbers_{'high' if sum(numbers) > 20 else 'low'}",
            f"max_number_{max(numbers)}",
            f"has_one_{1 in numbers}"
        ]
        
        for feature in features:
            self.pressure_correlations[feature] = (
                self.pressure_correlations[feature] * 0.8 + actual_pressure * 0.2
            )

    def _learn_rules(self, game_state, active_strand, is_success):
        """规则学习"""
        numbers = game_state['numbers']
        target = game_state['target']
        sorted_nums = sorted(numbers)
        
        patterns_to_learn = []
        
        if len(numbers) >= 2:
            top_two_sum = sorted_nums[-1] + sorted_nums[-2]
            top_two_prod = sorted_nums[-1] * sorted_nums[-2]
            patterns_to_learn.append(f"top2_sum_vs_target_{target>top_two_sum}")
            patterns_to_learn.append(f"top2_prod_vs_target_{target>top_two_prod}")
        
        if target % 2 == 0:
            patterns_to_learn.append(f"target_even")
        else:
            patterns_to_learn.append(f"target_odd")
        
        if any(target % n == 0 for n in numbers if n != 0):
            patterns_to_learn.append(f"has_factor")
        
        for pattern in patterns_to_learn:
            self.rule_patterns[pattern]['success' if is_success else 'failure'] += 1
            self.rule_patterns[pattern]['total'] += 1
        
        segment_size = 5
        for i in range(0, len(active_strand) - segment_size + 1):
            segment = tuple(active_strand[i:i+segment_size])
            utility_change = 0.1 if is_success else -0.05
            self.gene_segment_utility[segment] = max(-1.0, min(1.0, 
                self.gene_segment_utility.get(segment, 0.0) + utility_change))
        
        if is_success and len(active_strand) >= 8:
            best_segment = None
            best_utility = -float('inf')
            
            for i in range(0, len(active_strand) - 5):
                segment = tuple(active_strand[i:i+5])
                utility = self.gene_segment_utility.get(segment, 0.0)
                if utility > best_utility:
                    best_utility = utility
                    best_segment = segment
            
            if best_segment and best_utility > 0.3:
                self._add_to_memory_pool(best_segment, best_utility)

    def _add_to_memory_pool(self, segment, utility):
        """添加基因片段到记忆池"""
        for i, (mem_seg, mem_util, count, time) in enumerate(self.memory_gene_pool):
            if mem_seg == segment:
                new_utility = (mem_util * count + utility) / (count + 1)
                self.memory_gene_pool[i] = (segment, new_utility, count + 1, self.total_attempts)
                return
        
        if len(self.memory_gene_pool) >= self.memory_capacity:
            self.memory_gene_pool.sort(key=lambda x: x[1])
            removed = self.memory_gene_pool.pop(0)
            print(f" [记忆管理] 移除低效用片段 {removed[0]} (效用值: {removed[1]:.2f})")
        
        self.memory_gene_pool.append((segment, utility, 1, self.total_attempts))
        print(f" [记忆管理] 添加新片段 {segment} (效用值: {utility:.2f})")

    def _apply_learned_knowledge(self, game_state):
        """应用学到的知识"""
        numbers = game_state['numbers']
        target = game_state['target']
        sorted_nums = sorted(numbers)
        
        default_ops = ['*', '/', '+', '-']
        
        rules_to_check = []
        
        if len(numbers) >= 2:
            top_two_prod = sorted_nums[-1] * sorted_nums[-2]
            if target > top_two_prod * 1.5:
                rules_to_check.append(("target_much_larger", ['+', '*', '-', '/']))
            elif target < sorted_nums[0] * 2:
                rules_to_check.append(("target_small", ['-', '/', '+', '*']))
        
        if any(target % n == 0 for n in numbers if n != 0):
            rules_to_check.append(("has_factor", ['*', '/', '+', '-']))
        
        for rule in ['target_even', 'target_odd', 'has_factor']:
            if rule in self.rule_patterns:
                success_rate = self.rule_patterns[rule]['success'] / max(1, self.rule_patterns[rule]['total'])
                if success_rate > 0.6:
                    if rule == 'target_even':
                        rules_to_check.append((rule, ['*', '+', '-', '/']))
                    elif rule == 'has_factor':
                        rules_to_check.append((rule, ['*', '/', '+', '-']))
        
        if rules_to_check:
            rule_name, ops = rules_to_check[0]
            print(f" [规则应用] 使用规则 '{rule_name}'，操作符优先级: {ops}")
            return ops
        
        return default_ops

    def _express_decision(self, active_strand, game_state, is_doubt=False):
        """基因表达 - 生成表达式"""
        numbers = game_state['numbers']
        target = game_state['target']
        
        # 如果是怀疑模式，使用怀疑基因链的思考方式
        if is_doubt:
            ops_priority = self._apply_doubt_logic(game_state, active_strand)
        else:
            ops_priority = self._apply_learned_knowledge(game_state)
        
        best_expr = f"{numbers[0]}"
        best_score = float('inf')
        
        # 简单组合
        simple_combinations = []
        for i in range(len(numbers)):
            for j in range(i+1, len(numbers)):
                a, b = numbers[i], numbers[j]
                for op, func in [('+', lambda x,y: x+y), 
                                 ('-', lambda x,y: x-y),
                                 ('*', lambda x,y: x*y),
                                 ('/', lambda x,y: x/y if y!=0 else float('inf'))]:
                    try:
                        result = func(a, b)
                        distance = abs(result - target)
                        simple_combinations.append((distance, f"({a}{op}{b})", result))
                    except:
                        continue
        
        # 优先考虑简单组合
        simple_combinations.sort(key=lambda x: x[0])
        if simple_combinations and simple_combinations[0][0] < 5:
            base_expr = simple_combinations[0][1]
            remaining_nums = [n for n in numbers if str(n) not in base_expr]
            
            if not remaining_nums:
                return base_expr
            
            for perm in itertools.permutations(remaining_nums):
                for op_combo in itertools.product(ops_priority, repeat=len(perm)-1):
                    expr = base_expr
                    for i, num in enumerate(perm):
                        expr += op_combo[i-1] if i>0 else ''
                        expr += str(num)
                    
                    try:
                        val = eval(expr)
                        dist = abs(val - target)
                        if dist < best_score:
                            best_score = dist
                            best_expr = expr
                    except:
                        continue
        
        # 标准搜索
        for p in itertools.permutations(numbers):
            for op_combo in itertools.product(ops_priority, repeat=len(p)-1):
                expr_parts = []
                for i, num in enumerate(p):
                    expr_parts.append(str(num))
                    if i < len(op_combo):
                        expr_parts.append(op_combo[i])
                expr_str = "".join(expr_parts)
                
                try:
                    val = eval(expr_str)
                    dist_to_target = abs(val - target)
                    if dist_to_target < best_score:
                        best_score = dist_to_target
                        best_expr = expr_str
                except:
                    continue
        
        return best_expr

    def _apply_doubt_logic(self, game_state, doubt_strand):
        """应用怀疑逻辑 - 反着推理"""
        numbers = game_state['numbers']
        target = game_state['target']
        
        # 分析怀疑基因链的特征
        doubt_counts = Counter(doubt_strand[:10])  # 看前10个碱基
        
        # 怀疑逻辑：反着来！
        # 如果通常乘法优先，怀疑时就除法优先
        # 如果通常从大数开始，怀疑时就从小数开始
        
        # 默认操作符优先级
        normal_ops = ['*', '/', '+', '-']
        
        # 根据怀疑基因链调整
        if doubt_counts.get('X', 0) > 2:  # 探索基因多
            # 怀疑时尝试完全不同的顺序
            ops = normal_ops.copy()
            random.shuffle(ops)
            print(f"  [怀疑逻辑] 探索模式，随机操作符顺序: {ops}")
            return ops
            
        elif doubt_counts.get('C', 0) > 2:  # 保守基因多
            # 怀疑时更保守：减法和除法优先
            print(f"  [怀疑逻辑] 保守模式，减法和除法优先")
            return ['-', '/', '+', '*']
            
        elif doubt_counts.get('A', 0) > 2:  # 激进基因多
            # 怀疑时更激进：加法和乘法优先
            print(f"  [怀疑逻辑] 激进模式，加法和乘法优先")
            return ['+', '*', '-', '/']
            
        elif doubt_counts.get('T', 0) > 2:  # 灵活基因多
            # 怀疑时尝试目标除以数字
            print(f"  [怀疑逻辑] 灵活模式，尝试目标除以数字")
            return ['/', '*', '+', '-']
            
        else:
            # 默认怀疑：反着来！
            print(f"  [怀疑逻辑] 反转常规顺序")
            return list(reversed(normal_ops))

    def _generate_doubt_signal(self, game_state, initial_action):
        """生成怀疑信号 - 基于多个因素判断"""
        numbers = game_state['numbers']
        target = game_state['target']
        
        # 1. 检查表达式复杂度
        complexity_score = 0
        if initial_action:
            # 粗略估计表达式复杂度
            complexity_score = len(initial_action.replace(' ', '')) / 10
        
        # 2. 检查目标可达性
        reachability_score = 0
        sorted_nums = sorted(numbers, reverse=True)
        if len(sorted_nums) >= 2:
            max_possible = sorted_nums[0] * sorted_nums[1]
            if target > max_possible:
                reachability_score = 0.3
        
        # 3. 检查近期表现
        performance_score = 0
        if len(self.performance_history) > 5:
            recent_failures = sum(1 for x in self.performance_history[-5:] if x == 0)
            performance_score = recent_failures / 5
        
        # 4. 检查连续失败
        streak_score = self.failure_streak * 0.1
        
        # 5. 怀疑基因链活跃度
        doubt_gene_score = sum(1 for base in self.doubt_gene_chain['primary'][:10] 
                              if base in ['X', 'T']) / 10
        
        # 综合怀疑分数
        doubt_score = (
            complexity_score * 0.2 +
            reachability_score * 0.2 +
            performance_score * 0.3 +
            streak_score * 0.2 +
            doubt_gene_score * 0.1
        )
        
        # 决定是否怀疑
        should_doubt = random.random() < doubt_score
        
        # 怀疑原因
        reasons = []
        if complexity_score > 0.1:
            reasons.append("表达式复杂")
        if reachability_score > 0.1:
            reasons.append("目标难达到")
        if performance_score > 0.2:
            reasons.append("近期表现差")
        if streak_score > 0.1:
            reasons.append(f"连续失败{self.failure_streak}次")
        if doubt_gene_score > 0.1:
            reasons.append("怀疑基因活跃")
            
        reason_str = " & ".join(reasons) if reasons else "随机检查"
        
        return should_doubt, reason_str, doubt_score

    def _doubt_rethinking(self, initial_action, game_state):
        """怀疑重新思考 - 用怀疑基因链反着推理"""
        numbers = game_state['numbers']
        target = game_state['target']
        
        print(f"  [怀疑思考] 开始反向推理...")
        
        # 方法1：尝试相反的操作符优先级
        print(f"  [怀疑思考] 方法1: 尝试相反的操作符顺序")
        ops_reversed = ['-', '/', '+', '*']  # 与常规相反
        result1 = self._search_with_ops(numbers, target, ops_reversed)
        
        # 方法2：尝试从目标往回推
        print(f"  [怀疑思考] 方法2: 从目标往回推")
        result2 = self._search_from_target(numbers, target)
        
        # 方法3：尝试怀疑基因链的逻辑
        print(f"  [怀疑思考] 方法3: 使用怀疑基因链推理")
        doubt_ops = self._apply_doubt_logic(game_state, self.doubt_gene_chain['primary'])
        result3 = self._search_with_ops(numbers, target, doubt_ops)
        
        # 评估所有结果
        candidates = [
            (result1, self._evaluate_expression(result1, target, numbers)),
            (result2, self._evaluate_expression(result2, target, numbers)),
            (result3, self._evaluate_expression(result3, target, numbers)),
            (initial_action, self._evaluate_expression(initial_action, target, numbers))
        ]
        
        # 过滤掉无效的表达式（值为None或分数为无穷大）
        valid_candidates = [(expr, eval_info) for expr, eval_info in candidates 
                           if expr and eval_info['value'] is not None and eval_info['score'] < float('inf')]
        
        if not valid_candidates:
            print(f"  [怀疑思考] 所有候选都无效，保留原始表达式")
            return initial_action
        
        # 选择最好的
        valid_candidates.sort(key=lambda x: x[1]['score'])
        best_expr, best_eval = valid_candidates[0]
        
        print(f"  [怀疑思考] 最佳候选: {best_expr} (分数: {best_eval['score']:.2f})")
        
        return best_expr

    def _search_with_ops(self, numbers, target, ops_priority):
        """用特定操作符优先级搜索"""
        if not numbers:  # 处理空列表的情况
            return ""
            
        best_expr = f"{numbers[0]}"
        best_score = float('inf')
        
        # 限制搜索数量
        max_permutations = min(30, math.factorial(len(numbers)))
        
        perm_count = 0
        for p in itertools.permutations(numbers):
            if perm_count >= max_permutations:
                break
            perm_count += 1
            
            for op_combo in itertools.product(ops_priority, repeat=len(p)-1):
                expr_parts = []
                for i, num in enumerate(p):
                    expr_parts.append(str(num))
                    if i < len(op_combo):
                        expr_parts.append(op_combo[i])
                expr_str = "".join(expr_parts)
                
                try:
                    val = eval(expr_str)
                    dist_to_target = abs(val - target)
                    if dist_to_target < best_score:
                        best_score = dist_to_target
                        best_expr = expr_str
                except:
                    continue
        
        return best_expr

    def _search_from_target(self, numbers, target):
        """从目标往回推"""
        # 尝试用目标除以某个数字
        for num in numbers:
            if num != 0 and target % num == 0:
                result = target // num
                remaining = [n for n in numbers if n != num]
                # 尝试用剩余数字得到result
                expr = self._search_with_ops(remaining, result, ['+', '-', '*', '/'])
                if expr and expr != "" and expr != str(result) if remaining else False:
                    return f"({expr})*{num}"
        
        # 尝试目标加减某个数字
        for num in numbers:
            remaining = [n for n in numbers if n != num]
            # 尝试用剩余数字得到target-num
            expr1 = self._search_with_ops(remaining, target - num, ['+', '-', '*', '/'])
            if expr1 and expr1 != "" and expr1 != str(target - num) if remaining else False:
                return f"{expr1}+{num}"
            
            # 尝试用剩余数字得到target+num
            expr2 = self._search_with_ops(remaining, target + num, ['+', '-', '*', '/'])
            if expr2 and expr2 != "" and expr2 != str(target + num) if remaining else False:
                return f"{expr2}-{num}"
        
        # 如果所有方法都失败，返回一个简单的表达式
        return f"{numbers[0]}" if numbers else ""

    def _evaluate_expression(self, expr, target, numbers):
        """评估表达式"""
        if not expr:  # 处理空表达式
            return {'value': None, 'distance': float('inf'), 'complexity': 100, 
                    'completeness': 0, 'score': float('inf')}
            
        try:
            val = eval(expr)
            distance = abs(val - target)
            complexity = len(expr.replace(' ', ''))
            
            # 检查是否使用了所有数字
            import re
            used_nums = []
            for match in re.finditer(r'\d+\.?\d*', expr):
                used_nums.append(float(match.group()))
            
            completeness = 1.0 if sorted(used_nums) == sorted(numbers) else 0.5
            
            # 综合分数（越低越好）
            score = distance * 0.7 + complexity * 0.2 - completeness * 0.1
            
            return {
                'value': val,
                'distance': distance,
                'complexity': complexity,
                'completeness': completeness,
                'score': score
            }
        except:
            return {'value': None, 'distance': float('inf'), 'complexity': 100, 
                    'completeness': 0, 'score': float('inf')}

    def _update_strategy_mode(self):
        """更新策略模式"""
        if len(self.performance_history) < 10:
            return
        
        recent_success_rate = sum(self.performance_history[-10:]) / 10
        
        if self.total_attempts > 0:
            self.recent_success_rate = self.success_count / self.total_attempts
        
        if recent_success_rate < 0.15:
            self.strategy_mode = "explorative"
            print(f" [策略切换] 切换到探索模式 (近期成功率: {recent_success_rate:.2f})")
        elif recent_success_rate > 0.4:
            self.strategy_mode = "exploitative"
            print(f" [策略切换] 切换到开发模式 (近期成功率: {recent_success_rate:.2f})")
        else:
            self.strategy_mode = "balanced"

    def act(self, game_state):
        """执行一次行动"""
        self.total_attempts += 1
        print(f"\n--- 第 {self.total_attempts} 局 ---")
        print(f"目标数字: {game_state['target']}")
        print(f"可用数字: {game_state['numbers']}")

        # 更新策略模式
        self._update_strategy_mode()
        
        pressure = self._calculate_pressure(game_state)
        game_state_key = (tuple(sorted(game_state['numbers'])), game_state['target'])
        
        # 步骤1: 初步决策
        active_strand = self._get_active_strand(pressure, game_state_key)
        initial_action = self._express_decision(active_strand, game_state, is_doubt=False)
        print(f"GEDA 初步生成的表达式: {initial_action}")
        
        # 步骤2: 怀疑判断
        should_doubt, doubt_reason, doubt_score = self._generate_doubt_signal(game_state, initial_action)
        
        print(f"  [怀疑分析] 怀疑分数: {doubt_score:.2f}, 原因: {doubt_reason}")
        print(f"  [怀疑分析] 是否怀疑: {'是' if should_doubt else '否'}")
        
        final_action = initial_action
        was_doubt_helpful = False
        
        if should_doubt:
            print(f"  [自我怀疑] 触发，开始反向推理...")
            print(f"  [怀疑基因链] {''.join(self.doubt_gene_chain['primary'][:10])}...")
            
            # 使用怀疑重新思考
            doubted_action = self._doubt_rethinking(initial_action, game_state)
            
            print(f"  [自我怀疑] 反向推理结果: {doubted_action}")
            
            # 评估两个表达式
            first_eval = self._evaluate_expression(initial_action, game_state['target'], game_state['numbers'])
            second_eval = self._evaluate_expression(doubted_action, game_state['target'], game_state['numbers'])
            
            # 选择更好的
            if second_eval['score'] < first_eval['score']:
                final_action = doubted_action
                was_doubt_helpful = True
                print(f"  [自我怀疑] ✅ 反向推理改进了解 (分数: {first_eval['score']:.2f} -> {second_eval['score']:.2f})")
            else:
                was_doubt_helpful = False
                print(f"  [自我怀疑] ❌ 反向推理未改进 (分数: {first_eval['score']:.2f} vs {second_eval['score']:.2f})")
            
            # 记录怀疑效果
            self.doubt_states.append({
                'was_helpful': was_doubt_helpful,
                'original_score': first_eval['score'],
                'doubted_score': second_eval['score'],
                'reason': doubt_reason
            })
            
            # 更新怀疑有效性
            if len(self.doubt_states) > 0:
                helpful_count = sum(1 for state in self.doubt_states[-10:] if state['was_helpful'])
                total_count = min(10, len(self.doubt_states))
                self.doubt_effectiveness = helpful_count / total_count if total_count > 0 else 0.5
                print(f"  [怀疑统计] 近期有效性: {self.doubt_effectiveness:.1%}")

        print(f"GEDA 最终生成的表达式: {final_action}")

        # 验证结果
        is_success = self._validate_expression(final_action, game_state['target'], game_state['numbers'])
        
        # 更新表现记录
        if is_success:
            self.success_streak += 1
            self.failure_streak = 0
            self.success_count += 1
            print("结果: \033[92m成功!\033[0m 表达式成立。")
            
            key = (tuple(game_state['numbers']), game_state['target'])
            self.memory_dbs['success'][key].append(final_action)
        else:
            self.failure_streak += 1
            self.success_streak = 0
            print("结果: \033[91m失败!\033[0m 表达式不成立或非法。")
            
            key = (tuple(game_state['numbers']), game_state['target'])
            self.memory_dbs['failure'][key].append(final_action)
        
        # 记录本次表现
        self.performance_history.append(1 if is_success else 0)
        if len(self.performance_history) > self.history_window:
            self.performance_history.pop(0)
        
        # 学习与更新
        self._analyze_game_state_for_pressure(game_state, pressure)
        self._learn_rules(game_state, active_strand, is_success)
        
        # 定期变异
        if self.total_attempts % 15 == 0:
            self._mutate_gene_chain()
            self._mutate_doubt_gene_chain()
        
        # 记忆库清理
        if self.total_attempts % 50 == 0:
            self._cleanup_memory()
            
        return is_success

    def _mutate_gene_chain(self):
        """主基因链变异"""
        chain = self.gene_chain['primary']
        
        mutation_type = random.choice(['replace', 'insert', 'delete'])
        
        if mutation_type == 'insert' and len(chain) < 30:
            insertion_point = random.randint(0, len(chain))
            new_base = random.choice(self.bases)
            chain = chain[:insertion_point] + [new_base] + chain[insertion_point:]
                
        elif mutation_type == 'replace' and chain:
            replace_point = random.randint(0, len(chain)-1)
            new_base = random.choice([b for b in self.bases if b != chain[replace_point]])
            chain[replace_point] = new_base
                
        elif mutation_type == 'delete' and len(chain) > 10:
            delete_point = random.randint(0, len(chain)-1)
            chain = chain[:delete_point] + chain[delete_point+1:]
        
        self.gene_chain['primary'] = chain
        self.gene_chain['complementary'] = [self.complement_map[b] for b in chain]
        
        print(f"[系统日志] 主基因链变异，新长度: {len(chain)}")

    def _mutate_doubt_gene_chain(self):
        """怀疑基因链变异"""
        chain = self.doubt_gene_chain['primary']
        
        # 怀疑基因链变异更频繁
        for _ in range(2):  # 变异两次
            mutation_type = random.choice(['replace', 'insert', 'delete'])
            
            if mutation_type == 'insert' and len(chain) < 20:
                insertion_point = random.randint(0, len(chain))
                # 怀疑链更倾向于X和T
                weights = {'A': 0.1, 'G': 0.2, 'C': 0.2, 'T': 0.3, 'X': 0.2}
                new_base = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
                chain = chain[:insertion_point] + [new_base] + chain[insertion_point:]
                    
            elif mutation_type == 'replace' and chain:
                replace_point = random.randint(0, len(chain)-1)
                weights = {'A': 0.1, 'G': 0.2, 'C': 0.2, 'T': 0.3, 'X': 0.2}
                new_base = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
                chain[replace_point] = new_base
                    
            elif mutation_type == 'delete' and len(chain) > 8:
                delete_point = random.randint(0, len(chain)-1)
                chain = chain[:delete_point] + chain[delete_point+1:]
        
        self.doubt_gene_chain['primary'] = chain
        self.doubt_gene_chain['complementary'] = [self.complement_map[b] for b in chain]
        
        print(f"[系统日志] 怀疑基因链变异，新链: {''.join(chain[:10])}...")

    def _cleanup_memory(self):
        """清理记忆库"""
        if len(self.memory_gene_pool) > self.memory_capacity * 0.8:
            self.memory_gene_pool.sort(key=lambda x: x[1])
            to_remove = len(self.memory_gene_pool) - int(self.memory_capacity * 0.7)
            if to_remove > 0:
                self.memory_gene_pool = self.memory_gene_pool[to_remove:]
                print(f" [记忆清理] 移除了 {to_remove} 个低效用记忆片段")
        
        rules_to_remove = []
        for rule, data in self.rule_patterns.items():
            if data['total'] < 3 and self.total_attempts > 20:
                rules_to_remove.append(rule)
        
        for rule in rules_to_remove:
            del self.rule_patterns[rule]
        
        if rules_to_remove:
            print(f" [规则清理] 移除了 {len(rules_to_remove)} 个很少使用的规则")

def run_optimized_math_game(agent, num_games=100):
    """运行优化版数学游戏"""
    print(f"=== 优化版宏观正反馈 GEDA 数学游戏开始 (共 {num_games} 局) ===")
    print(f"初始策略模式: {agent.strategy_mode}")
    print(f"初始怀疑有效性: {agent.doubt_effectiveness:.1%}")
    
    for i in range(num_games):
        if i < 30:
            target = random.randint(15, 40)
            num_count = random.randint(3, 4)
        elif i < 70:
            target = random.randint(20, 50)
            num_count = random.randint(4, 5)
        else:
            target = random.randint(25, 60)
            num_count = random.randint(4, 5)
        
        numbers = [random.randint(1, 10) for _ in range(num_count)]
        
        game_state = {'target': target, 'numbers': numbers}
        agent.act(game_state)
    
    print(f"\n=== 优化版游戏结束 (共 {num_games} 局) ===")

def generate_optimized_intelligence_report(agent):
    """生成优化版智能判断报告"""
    success_rate = (agent.success_count / agent.total_attempts) * 100 if agent.total_attempts > 0 else 0
    
    print("\n" + "="*50)
    print("          优化版宏观正反馈 GEDA 代理智能判断报告")
    print("="*50)
    print(f"总游戏轮次: {agent.total_attempts}")
    print(f"成功次数: {agent.success_count}")
    print(f"失败次数: {agent.total_attempts - agent.success_count}")
    print(f"【最终成功率】: {success_rate:.2f}%")
    print(f"当前策略模式: {agent.strategy_mode}")
    print(f"怀疑有效性: {agent.doubt_effectiveness:.1%}")
    
    if agent.total_attempts == 0:
        print("\n分析: 尚未进行任何游戏，无法评估。")
        return
    
    # 计算近期表现
    recent_performance = "未知"
    if len(agent.performance_history) >= 10:
        recent_success = sum(agent.performance_history[-10:])
        recent_performance = f"{recent_success/10*100:.1f}%"
    
    print(f"\n--- 智能行为分析 ---")
    print(f"- **近期表现** (最近10局): {recent_performance}")
    
    # 怀疑机制分析
    if agent.doubt_states:
        total_doubts = len(agent.doubt_states)
        helpful_doubts = sum(1 for state in agent.doubt_states if state['was_helpful'])
        print(f"- **怀疑机制**: 触发 {total_doubts} 次，有帮助 {helpful_doubts} 次 ({helpful_doubts/total_doubts*100:.1f}%)")
        
        # 分析怀疑原因的有效性
        reason_effectiveness = {}
        for state in agent.doubt_states[-20:]:  # 看最近20次
            reason = state['reason']
            if reason not in reason_effectiveness:
                reason_effectiveness[reason] = {'helpful': 0, 'total': 0}
            reason_effectiveness[reason]['total'] += 1
            if state['was_helpful']:
                reason_effectiveness[reason]['helpful'] += 1
        
        if reason_effectiveness:
            print(f"- **怀疑原因分析**:")
            for reason, stats in list(reason_effectiveness.items())[:3]:  # 显示前3个
                rate = stats['helpful'] / stats['total'] if stats['total'] > 0 else 0
                print(f"  '{reason}': {stats['helpful']}/{stats['total']} ({rate:.1%})")
    
    if agent.success_count > 0:
        if success_rate > 40:
            print("- **元学习能力**: 卓越。怀疑机制有效改进了决策。")
        elif success_rate > 30:
            print("- **元学习能力**: 优秀。系统展现出良好的自我反思能力。")
        elif success_rate > 20:
            print("- **元学习能力**: 良好。怀疑机制开始发挥作用。")
        else:
            print("- **元学习能力**: 一般。怀疑机制效果有限。")
    else:
        print("- **元学习能力**: 待观察。需要更多游戏轮次。")
    
    # 记忆系统分析
    if agent.memory_gene_pool:
        avg_utility = sum([util for _, util, _, _ in agent.memory_gene_pool]) / len(agent.memory_gene_pool)
        print(f"- **记忆系统**: 记忆池中有 {len(agent.memory_gene_pool)} 个片段，平均效用值: {avg_utility:.2f}")
    else:
        print("- **记忆系统**: 记忆池为空，系统仍在初始学习阶段。")
    
    # 规则学习分析
    if agent.rule_patterns:
        high_value_rules = [r for r, d in agent.rule_patterns.items() 
                          if d['total'] >= 3 and d['success']/d['total'] > 0.6]
        print(f"- **规则学习**: 发现了 {len(high_value_rules)} 个高价值规则")
        if high_value_rules:
            print(f"  示例: {high_value_rules[:3]}")
    
    print("\n--- 系统状态 ---")
    print(f"主基因链长度: {len(agent.gene_chain['primary'])}")
    print(f"怀疑基因链: {''.join(agent.doubt_gene_chain['primary'][:10])}...")
    print(f"连续成功: {agent.success_streak}")
    print(f"连续失败: {agent.failure_streak}")
    print(f"记忆池利用率: {len(agent.memory_gene_pool)}/{agent.memory_capacity}")
    
    print("\n--- 总体评价 ---")
    if success_rate >= 45:
        print("系统展现了顶级的智能水平。怀疑机制有效地实现了反向推理和自我改进。")
    elif success_rate >= 35:
        print("系统表现出卓越的智能特征。专门的怀疑基因链带来了有效的反思能力。")
    elif success_rate >= 25:
        print("系统具备良好的智能基础。反向推理机制开始展现价值。")
    elif success_rate > 0:
        print("系统展现出学习能力，怀疑机制需要更多时间优化。")
    else:
        print("系统尚未展现出学习效果。")
    
    print("="*50)


# --- 主程序 ---
if __name__ == "__main__":
    print("="*60)
    print("     优化版宏观正反馈与记忆基因 GEDA 代理")
    print("     专门怀疑基因链 + 反向推理机制")
    print("="*60)
    
    # 创建代理
    my_optimized_agent = OptimizedMacroReinforcedGEDA()
    
    # 运行100局游戏
    run_optimized_math_game(my_optimized_agent, num_games=100)
    
    # 生成详细智能报告
    generate_optimized_intelligence_report(my_optimized_agent)
