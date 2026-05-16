import pygame
import numpy as np
import random
import math
import os
from datetime import datetime
from collections import defaultdict, deque

# ========== 贪吃蛇环境（完全可复现） ==========
class SnakeEnv:
    ACTIONS = [0, 1, 2, 3]          # 上,下,左,右
    ACTION_DELTA = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
    ACTION_NAME = ['U', 'D', 'L', 'R']

    def __init__(self, width=10, height=10, cell_size=40):
        self.w, self.h = width, height
        self.cell_size = cell_size
        self.reset()

    def reset(self):
        self.snake = [(self.w//2, self.h//2), (self.w//2-1, self.h//2), (self.w//2-2, self.h//2)]
        self.direction = 3
        self._place_food()
        self.score = 0
        self.done = False
        self.steps = 0
        return self._get_state(), self.snake[0], self.food

    def _place_food(self):
        while True:
            pos = (random.randint(0, self.w-1), random.randint(0, self.h-1))
            if pos not in self.snake:
                self.food = pos
                break

    def _get_state(self):
        head = self.snake[0]
        bx = head[0] * 5 // self.w
        by = head[1] * 5 // self.h
        obs = []
        for act in self.ACTIONS:
            dx, dy = self.ACTION_DELTA[act]
            nx, ny = head[0]+dx, head[1]+dy
            if nx < 0 or nx >= self.w or ny < 0 or ny >= self.h or (nx, ny) in self.snake:
                obs.append('1')
            else:
                obs.append('0')
        obs_code = ''.join(obs)
        fx, fy = self.food
        hx, hy = head
        food_dir = ['0','0','0','0']
        if fx < hx: food_dir[2] = '1'
        if fx > hx: food_dir[3] = '1'
        if fy < hy: food_dir[0] = '1'
        if fy > hy: food_dir[1] = '1'
        food_code = ''.join(food_dir)
        return f"{bx}{by}|{obs_code}|{food_code}"

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, self.snake[0], self.food
        self.steps += 1
        head = self.snake[0]
        dx, dy = self.ACTION_DELTA[action]
        new_head = (head[0]+dx, head[1]+dy)

        old_dist = abs(head[0]-self.food[0]) + abs(head[1]-self.food[1])
        new_dist = abs(new_head[0]-self.food[0]) + abs(new_head[1]-self.food[1])

        self.snake.insert(0, new_head)

        reward = 0
        if (new_head in self.snake[1:] or
            new_head[0] < 0 or new_head[0] >= self.w or
            new_head[1] < 0 or new_head[1] >= self.h):
            self.done = True
            reward = -10
        else:
            if new_head == self.food:
                self.score += 1
                reward = 10
                self._place_food()
            else:
                self.snake.pop()
                delta = old_dist - new_dist
                reward = max(-0.5, min(0.5, delta * 0.3))

        if action in self.ACTIONS:
            self.direction = action

        return self._get_state(), reward, self.done, self.snake[0], self.food

    # ---------- 渲染（仅用于重放） ----------
    def draw(self, screen, font):
        screen.fill((30,30,30))
        for x in range(self.w+1):
            pygame.draw.line(screen, (60,60,60), (x*self.cell_size,0),
                             (x*self.cell_size, self.h*self.cell_size), 1)
        for y in range(self.h+1):
            pygame.draw.line(screen, (60,60,60), (0, y*self.cell_size),
                             (self.w*self.cell_size, y*self.cell_size), 1)
        fx, fy = self.food
        pygame.draw.rect(screen, (255,100,100),
                         (fx*self.cell_size+2, fy*self.cell_size+2,
                          self.cell_size-4, self.cell_size-4))
        for i, seg in enumerate(self.snake):
            color = (0,200,0) if i==0 else (0,150,0)
            pygame.draw.rect(screen, color,
                             (seg[0]*self.cell_size+2, seg[1]*self.cell_size+2,
                              self.cell_size-4, self.cell_size-4))
        score_surf = font.render(f"Score: {self.score}", True, (255,255,255))
        screen.blit(score_surf, (10,10))
        pygame.display.flip()


# ========== GEDA 实时感知模块 ==========
class GEDA:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.vision_range = 4
        self.vision_depth = 3
        self.cdrcc_waves = self._init_waves()
        self.yanghui = self._generate_yanghui(25)
        self.gene_chain = self._init_gene()
        self.total_steps = 0

    def _init_waves(self):
        waves = []
        for d in ['U','D','L','R']:
            waves.append({
                'direction': d,
                'amplitude': 0.7 + random.random()*0.5,
                'frequency': 0.5 + random.random()*0.4,
                'phase': random.random()*math.pi*2,
                'base_value': 0.6,
                'success_score': 0.5
            })
        return waves

    def _generate_yanghui(self, n):
        tri = []
        for i in range(n):
            row = [1]*(i+1)
            for j in range(1,i):
                row[j] = tri[i-1][j-1] + tri[i-1][j]
            tri.append(row)
        mat = []
        for i,row in enumerate(tri):
            mat.append(row + [0]*(n-len(row)))
        return mat

    def _init_gene(self):
        primary = ['A','G','C','T','M']*4
        comp = {'A':'T','T':'A','G':'C','C':'G','M':'M'}
        complementary = [comp[b] for b in primary]
        return {'primary': primary, 'complementary': complementary}

    def vision_scores(self, head, food, snake):
        scores = [0.0, 0.0, 0.0, 0.0]
        dx, dy = food[0]-head[0], food[1]-head[1]
        if abs(dx) > abs(dy):
            if dx > 0:
                scores[3] += 0.8
                for i in range(1, min(self.vision_range, dx)):
                    if (head[0]+i, head[1]) in snake:
                        scores[3] -= 0.3*(self.vision_range-i)/self.vision_range
                        break
            else:
                scores[2] += 0.8
                for i in range(1, min(self.vision_range, -dx)):
                    if (head[0]-i, head[1]) in snake:
                        scores[2] -= 0.3*(self.vision_range-i)/self.vision_range
                        break
        else:
            if dy > 0:
                scores[1] += 0.8
                for i in range(1, min(self.vision_range, dy)):
                    if (head[0], head[1]+i) in snake:
                        scores[1] -= 0.3*(self.vision_range-i)/self.vision_range
                        break
            else:
                scores[0] += 0.8
                for i in range(1, min(self.vision_range, -dy)):
                    if (head[0], head[1]-i) in snake:
                        scores[0] -= 0.3*(self.vision_range-i)/self.vision_range
                        break

        dir_delta = [(0,-1),(0,1),(-1,0),(1,0)]
        for a in range(4):
            dx, dy = dir_delta[a]
            safe_cells = 0
            has_food = False
            food_dist = 0
            for dist in range(1, self.vision_depth+1):
                check = (head[0]+dx*dist, head[1]+dy*dist)
                if not (0 <= check[0] < self.grid_size and 0 <= check[1] < self.grid_size):
                    scores[a] -= 0.2*(self.vision_depth-dist+1)/self.vision_depth
                    break
                if check in snake:
                    scores[a] -= 0.3*(self.vision_depth-dist+1)/self.vision_depth
                    break
                if check == food:
                    has_food = True
                    food_dist = dist
                safe_cells += 1
            if safe_cells == self.vision_depth:
                scores[a] += 0.3
            if has_food:
                scores[a] += 0.5 / food_dist
        return {0:scores[0],1:scores[1],2:scores[2],3:scores[3]}

    def cdrcc_scores(self):
        scores = [0.0,0.0,0.0,0.0]
        for wave in self.cdrcc_waves:
            idx = ['U','D','L','R'].index(wave['direction'])
            val = (wave['amplitude'] *
                   math.cos(wave['frequency'] * self.total_steps / 10 + wave['phase']) +
                   wave['base_value'])
            val *= wave['success_score']
            scores[idx] += val
        return {0:scores[0],1:scores[1],2:scores[2],3:scores[3]}

    def gene_influence(self, head, food):
        dx, dy = food[0]-head[0], food[1]-head[1]
        dist = abs(dx)+abs(dy)
        row = dist % len(self.yanghui)
        col = (head[0]+head[1]) % len(self.yanghui[0])
        yang = self.yanghui[row][col]
        gene_idx = int(yang) % len(self.gene_chain['primary'])
        base = self.gene_chain['primary'][gene_idx]
        effects = {
            'A':{0:0.3,3:0.2}, 'G':{1:0.3,2:0.2},
            'C':{2:0.3,0:0.2}, 'T':{3:0.3,1:0.2},
            'M':{0:0.1,1:0.1,2:0.1,3:0.1}
        }
        infl = {0:0.0,1:0.0,2:0.0,3:0.0}
        if base in effects:
            for a,val in effects[base].items():
                infl[a] += val
        return infl

    def get_base_scores(self, head, food, snake):
        v = self.vision_scores(head, food, snake)
        c = self.cdrcc_scores()
        g = self.gene_influence(head, food)
        scores = {}
        for a in range(4):
            scores[a] = v[a] + c[a] + g[a]*0.1
        return scores

    def update_wave_success(self, action, success):
        dir_name = ['U','D','L','R'][action]
        for wave in self.cdrcc_waves:
            if wave['direction'] == dir_name:
                if success:
                    wave['success_score'] = min(1.0, wave['success_score'] + 0.08)
                else:
                    wave['success_score'] = max(0.1, wave['success_score'] - 0.02)
                break


# ========== 经验记忆模块（RSNN置信度系统）==========
class ExperienceMemory:
    def __init__(self):
        self.memory = defaultdict(dict)
        self.trajectory = []
        self.death_trace = deque(maxlen=3)
        self.step_counter = 0

        # ----- 调优参数（针对高分瓶颈）-----
        self.step_alpha_close = 0.02      # 靠近/吃到食物时置信度增量
        self.step_alpha_far = 0.01        # 远离时置信度减量
        self.hunt_bonus = 0.5             # 成功路径强化值
        self.lateral_inhibition = 0.1     # 侧向抑制
        self.protect_steps = 500          # 永久边保护期
        self.perm_min_count = 5           # 固化最小尝试次数
        self.perm_min_conf = 0.7          # 固化最小置信度
        self.perm_protect = 300           # 固化后保护步数
        self.death_penalty = 0.8          # 死亡回溯惩罚因子
        self.demote_conf = 0.3            # 永久边降级阈值
        self.decay_interval = 500         # 衰减间隔（原300，延长避免过早衰减）
        self.decay_factor = 0.98          # 衰减因子（原0.95，减缓衰减）
        self.prune_min_trials = 10        # 剪枝最少尝试次数
        self.prune_conf = 0.3             # 剪枝置信度阈值
        self.prune_prob = 0.3             # 剪枝概率

    def _init_entry(self, state, action, conf=0.5):
        if action not in self.memory[state]:
            self.memory[state][action] = {
                'conf': conf,
                'type': 'temp',
                'count': 0,
                'last_used': self.step_counter,
                'protected_until': 0
            }

    def get_confidence(self, state, action):
        if state in self.memory and action in self.memory[state]:
            return self.memory[state][action]['conf']
        return 0.5

    def record_attempt(self, state, action):
        self._init_entry(state, action)
        e = self.memory[state][action]
        e['count'] += 1
        e['last_used'] = self.step_counter
        self.trajectory.append((state, action))
        if len(self.death_trace) == self.death_trace.maxlen:
            self.death_trace.popleft()
        self.death_trace.append((state, action))

    def learn_step(self, state, action, reward):
        self._init_entry(state, action)
        e = self.memory[state][action]
        if reward > 0.1:
            e['conf'] = min(0.95, e['conf'] + self.step_alpha_close)
        elif reward < -0.1:
            e['conf'] = max(0.1, e['conf'] - self.step_alpha_far)

        if e['type'] == 'temp' and e['count'] >= self.perm_min_count and e['conf'] >= self.perm_min_conf:
            e['type'] = 'perm'
            e['protected_until'] = self.step_counter + self.perm_protect

    def reinforce_path(self, path):
        for state, action in reversed(path):
            self._init_entry(state, action)
            e = self.memory[state][action]
            e['conf'] = min(0.95, e['conf'] + self.hunt_bonus)
            e['type'] = 'perm'
            e['protected_until'] = self.step_counter + self.protect_steps
            e['last_used'] = self.step_counter
            for a2, e2 in self.memory[state].items():
                if a2 != action:
                    e2['conf'] = max(0.1, e2['conf'] - self.lateral_inhibition)

    def record_death(self, trace):
        for state, action in trace:
            if state in self.memory and action in self.memory[state]:
                e = self.memory[state][action]
                e['conf'] *= self.death_penalty
                if e['type'] == 'perm' and e['conf'] < self.demote_conf:
                    e['type'] = 'temp'
                    e['protected_until'] = 0

    def decay_and_prune(self):
        self.step_counter += 1
        for state in list(self.memory.keys()):
            for action in list(self.memory[state].keys()):
                e = self.memory[state][action]
                if e['type'] == 'perm' and self.step_counter > e.get('protected_until', 0):
                    unused = self.step_counter - e.get('last_used', 0)
                    if unused >= self.decay_interval:
                        decay_times = unused // self.decay_interval
                        for _ in range(decay_times):
                            e['conf'] *= self.decay_factor
                        e['last_used'] += decay_times * self.decay_interval
                        if e['conf'] < self.demote_conf:
                            e['type'] = 'temp'
                            e['protected_until'] = 0
                if e['type'] == 'temp':
                    if e['count'] >= self.prune_min_trials and e['conf'] < self.prune_conf:
                        if random.random() < self.prune_prob:
                            del self.memory[state][action]
            if not self.memory[state]:
                del self.memory[state]

    def clear_trajectory(self):
        self.trajectory = []

    def get_high_confidence_bias(self, threshold=0.55, bias_value=0.35):
        """
        获取高置信度状态-动作的偏置，用于离线偏置模式
        - threshold: 置信度阈值（原0.7 → 0.55，让更多经验参与）
        - bias_value: 偏置强度（原0.2 → 0.35，让经验更有影响力）
        """
        bias = defaultdict(lambda: [0.0]*4)
        for state, actions in self.memory.items():
            for a, e in actions.items():
                if e['conf'] >= threshold:
                    bias[state][a] = bias_value
        return bias


# ========== 融合智能体：三种模式 ==========
class FusionAgent:
    def __init__(self, grid_size=10, modulation_strength=0.6, mode='geda_only'):
        """
        mode 可选:
            'geda_only'     - 纯 GEDA
            'positive_only' - 正向增益（只放大不压制）
            'offline_bias'  - 离线偏置（GEDA + 经验偏置，已大幅优化）
        """
        self.geda = GEDA(grid_size)
        self.exp = ExperienceMemory()
        self.beta = modulation_strength
        self.mode = mode
        self.bias_map = defaultdict(lambda: [0.0]*4)
        # 离线偏置模式的探索噪声（避免完全确定性）
        self.epsilon = 0.02 if mode == 'offline_bias' else 0.0

    def choose_action(self, state, head, food, snake_body):
        base = self.geda.get_base_scores(head, food, snake_body)

        if self.mode == 'geda_only':
            final = base.copy()

        elif self.mode == 'positive_only':
            final = {}
            for a in range(4):
                conf = self.exp.get_confidence(state, a)
                factor = 1 + self.beta * max(0, conf - 0.5)
                final[a] = base[a] * factor

        elif self.mode == 'offline_bias':
            final = base.copy()
            bias = self.bias_map[state]
            for a in range(4):
                final[a] += bias[a]      # 加偏置
            # 添加微小探索噪声
            if random.random() < self.epsilon:
                noise = {a: random.uniform(-0.1, 0.1) for a in range(4)}
                for a in range(4):
                    final[a] += noise[a]

        else:
            raise ValueError(f"未知模式: {self.mode}")

        # 安全过滤
        for a in range(4):
            dx, dy = SnakeEnv.ACTION_DELTA[a]
            nx, ny = head[0]+dx, head[1]+dy
            if (nx < 0 or nx >= 10 or ny < 0 or ny >= 10 or
                (nx, ny) in snake_body):
                final[a] = -100.0

        action = max(final.items(), key=lambda x: x[1])[0]

        self.exp.record_attempt(state, action)
        self.geda.total_steps += 1
        return action

    def learn(self, state, action, reward, done, ate_food=False):
        self.exp.learn_step(state, action, reward)
        success = (reward == 10)
        self.geda.update_wave_success(action, success)

        if ate_food:
            self.exp.reinforce_path(self.exp.trajectory)
            self.exp.clear_trajectory()
            # 【新增】吃到食物后立即更新偏置，让同一局后续动作也能受益
            if self.mode == 'offline_bias':
                self.load_bias()

        if done:
            self.exp.record_death(list(self.exp.death_trace))

    def decay_prune(self):
        self.exp.decay_and_prune()

    def clear_trajectory(self):
        self.exp.clear_trajectory()

    def load_bias(self, threshold=0.55, bias_value=0.35):
        """仅用于 offline_bias 模式：加载经验偏置（参数已优化）"""
        self.bias_map = self.exp.get_high_confidence_bias(threshold, bias_value)


# ========== 主程序：命令行选择 + 训练回合数输入 + 训练 + 重放截图 + Enter退出 ==========
def main():
    # ---------- 固定全局种子，保证可复现 ----------
    random.seed(42)
    np.random.seed(42)

    # ---------- 命令行模式选择 ----------
    print("\n" + "="*50)
    print("          RSNN-GEDA 贪吃蛇融合系统（优化版）")
    print("="*50)
    print("请选择运行模式：")
    print("  1. 纯 GEDA（经验完全不参与决策）")
    print("  2. 正向增益（经验只放大高置信度动作，不压制）")
    print("  3. 离线偏置（GEDA + 即时偏置 + 探索噪声）【推荐】")
    print("="*50)

    choice = input("请输入数字 (1/2/3) [默认: 3]: ").strip()
    if choice == '1':
        mode = 'geda_only'
        mode_name = "纯 GEDA"
    elif choice == '2':
        mode = 'positive_only'
        mode_name = "正向增益"
    else:
        mode = 'offline_bias'
        mode_name = "离线偏置（优化版）"

    # ---------- 训练回合数输入 ----------
    print("\n" + "-"*50)
    ep_input = input("请输入训练回合数 [默认: 400]: ").strip()
    if ep_input.isdigit():
        EPISODES = int(ep_input)
    else:
        EPISODES = 400
    print(f"训练回合数: {EPISODES}")
    print("-"*50)

    print(f"\n已选择模式：{mode_name}")
    print("开始训练...\n")

    # 参数设置
    W, H = 10, 10
    CELL_SIZE = 40

    env = SnakeEnv(W, H, CELL_SIZE)
    agent = FusionAgent(grid_size=W, modulation_strength=0.6, mode=mode)

    # ---------- 训练记录 ----------
    global_max_score = 0
    record_seed = None          # 最高分出现的那一局的随机种子
    record_actions = []        # 从开局到吃到最高分果实的完整动作序列

    for ep in range(EPISODES):
        random.seed(ep)
        state, head, food = env.reset()
        agent.clear_trajectory()

        if agent.mode == 'offline_bias':
            agent.load_bias()   # 开局加载一次偏置

        done = False
        actions_this_ep = []

        while not done:
            action = agent.choose_action(state, head, food, env.snake)
            actions_this_ep.append(action)
            next_state, reward, done, head, food = env.step(action)
            ate = (reward == 10)
            agent.learn(state, action, reward, done, ate)

            if ate and env.score > global_max_score:
                global_max_score = env.score
                record_seed = ep
                record_actions = actions_this_ep.copy()
                print(f"  新纪录！第 {ep+1} 回合，得分 {env.score}")

            state = next_state

        if ep % 5 == 0:
            agent.decay_prune()

    print(f"\n训练结束，全局最高分: {global_max_score}")

    # ---------- 如果没有得分，直接退出 ----------
    if record_seed is None:
        print("未吃到任何食物，无法截图。")
        input("\n按 Enter 键退出...")
        return

    # ---------- 初始化 Pygame，重放最高分一局并截图 ----------
    pygame.init()
    screen = pygame.display.set_mode((W*CELL_SIZE, H*CELL_SIZE + 50))
    pygame.display.set_caption(f"最高分重放 - 得分 {global_max_score} (模式: {mode_name})")
    font = pygame.font.SysFont('arial', 20)
    clock = pygame.time.Clock()

    random.seed(record_seed)
    env.reset()

    print(f"\n重放最高分一局（种子 {record_seed}），动作序列长度 {len(record_actions)}...")

    for i, action in enumerate(record_actions):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                input("\n按 Enter 键退出...")
                return

        env.draw(screen, font)
        _, reward, done, head, food = env.step(action)

        if reward == 10 and env.score == global_max_score:
            script_dir = os.path.dirname(__file__)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snake_max_score_{global_max_score}_{timestamp}.png"
            filepath = os.path.join(script_dir, filename)
            pygame.image.save(screen, filepath)
            print(f"✅ 截图已保存: {filename}")

        clock.tick(10)

    print("重放结束，窗口将保持，按 Enter 键关闭...")
    pygame.display.set_caption(f"最高分重放完成 - 得分 {global_max_score} (按 Enter 退出)")

    # ---------- 等待用户按 Enter 键退出 ----------
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                waiting = False
        pygame.display.flip()
        clock.tick(10)

    pygame.quit()
    print("程序已退出。")

if __name__ == "__main__":
    main()