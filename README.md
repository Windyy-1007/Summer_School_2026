# Summer School 2026 - Robot Reinforcement Learning

Repo này là một môi trường mô phỏng robot đi trên bản đồ ô vuông. Mục tiêu của
robot là:

1. Đi qua các checkpoint nếu bản đồ có checkpoint.
2. Đi tới đích cuối.
3. Dùng ít bước đi và ít lần quay nhất có thể.
4. Học cách tránh vật cản bằng Reinforcement Learning (RL).

Phần học sinh làm việc nhiều nhất là:

- `Algo/StudentRL.py`: vòng lặp Q-learning và cách chọn hành động.
- `rl_library.py`: thư viện hỗ trợ state, action, transition, reward, tính điểm,
  đọc bản đồ và lưu policy.

## 1. Chạy chương trình

### Yêu cầu

- Python 3.10 trở lên.
- Tkinter. Bản Python chính thức trên Windows thường đã có sẵn Tkinter.

Repo hiện không cần cài thêm package bằng `pip`.

### Chạy giao diện

Mở PowerShell tại thư mục gốc của repo:

```powershell
python .\main.py
```

Hoặc:

```powershell
.\runApp.ps1
```

Nếu PowerShell chặn file `.ps1`, dùng lệnh `python .\main.py`.

### Quy trình sử dụng đề xuất

1. Chạy `python .\main.py`.
2. Chọn số thứ tự map và bấm `Load Training Map`.
3. Chọn số episode, ví dụ `500`.
4. Bấm `Train Current Map` để học map đang mở.
5. Bấm `Run Student RL` để robot chạy bằng policy đã lưu.
6. Xem đường đi, reward, score và log trên giao diện.
7. Sửa thuật toán trong `Algo/StudentRL.py`, sau đó chạy lại để so sánh.

`Train All Maps` sẽ học lần lượt toàn bộ 30 map trong `maps/training/`. Các lần
train tiếp theo dùng tiếp Q-table đã lưu, không tự động học lại từ đầu.

Policy được lưu tại:

```text
maps/student_rl_policies/student_rl_policy.json
```

Bấm `Reset RL Model` nếu muốn xóa policy cũ và học lại từ đầu.

## 2. Điều khiển và chỉnh map

Phím điều khiển robot:

| Phím | Hành động |
| --- | --- |
| `W` | Đi thẳng |
| `S` | Đi lùi |
| `A` | Quay trái 90 độ |
| `D` | Quay phải 90 độ |

Các nút chính:

| Nút | Công dụng |
| --- | --- |
| `Show A* Reference` | Hiện đường đi tham khảo của A* |
| `Run Student RL` | Chạy policy Student RL đã lưu |
| `Reset Robot` | Đưa robot về vị trí bắt đầu |
| `Train Current Map` | Học thêm trên map hiện tại |
| `Train All Maps` | Học lần lượt tất cả training map |
| `Stop RL` | Dừng sau batch train hiện tại |
| `Reset RL Model` | Xóa Q-table đã lưu |
| `Save Map JSON` | Lưu map đang chỉnh |
| `Load Map JSON` | Mở một map JSON |

Trong `Map editor`, có thể đặt đích, checkpoint, vật cản tại node và vật cản
trên cạnh.

## 3. Bản đồ hoạt động như thế nào?

Mỗi vị trí có tọa độ `(x, y)`:

- `(0, 0)` nằm ở góc dưới bên trái.
- `x` tăng khi đi sang phải.
- `y` tăng khi đi lên trên.
- Robot chỉ đi giữa hai node cạnh nhau.

Có hai loại vật cản:

- **Blocked node**: ô mà robot không thể đứng vào.
- **Blocked edge**: đường nối giữa hai ô bị chặn.

Một file map JSON chứa các thông tin chính:

```json
{
  "width": 18,
  "height": 12,
  "start": [0, 0],
  "goal": [17, 11],
  "checkpoints": [[1, 10], [16, 1]],
  "blocked_nodes": [],
  "blocked_edges": []
}
```

Khi chạy thật trong mô phỏng, robot có hai bản đồ:

- `map`: bản đồ thật, chứa toàn bộ vật cản.
- `known_map`: phần bản đồ robot đã biết.

Robot phải di chuyển và phát hiện vật cản để cập nhật `known_map`.

## 4. Cách tính score

**Score là điểm kết quả cuối**, dùng để so sánh các đường chạy. Score không
phải là reward dùng để học Q-table.

Các giá trị:

| Thành phần | Điểm |
| --- | ---: |
| Tới đích cuối | `+400` |
| Mỗi checkpoint đã đi qua | `+200` |
| Mỗi bước tiến hoặc lùi thành công | `-4` |
| Mỗi lần quay trái hoặc phải | `-1` |

Nếu robot hoàn thành:

```text
score = 400
        + 200 * số checkpoint
        - 4 * số bước di chuyển
        - 1 * số lần quay
```

Ví dụ robot đi qua 2 checkpoint, dùng 10 bước và quay 3 lần:

```text
score = 400 + 2*200 - 10*4 - 3
      = 757
```

Nếu robot chưa hoàn thành:

```text
score = -1000 - khoảng cách Manhattan gần đích nhất đã đạt được
```

Ví dụ khoảng cách gần nhất tới đích là 4:

```text
score = -1000 - 4 = -1004
```

Khoảng cách Manhattan giữa `(x1, y1)` và `(x2, y2)` là:

```text
|x1 - x2| + |y1 - y2|
```

Lưu ý:

- Khi chưa hoàn thành, công thức score không cộng checkpoint và không trừ số
  bước/quay.
- Trong RL, chỉ đứng ở goal là chưa đủ. Episode chỉ kết thúc khi robot đã đi
  qua tất cả checkpoint rồi đứng tại goal.
- Score cao hơn là tốt hơn.

Hàm tính điểm là:

```python
calculate_score(
    num_moves,
    num_turns,
    checkpoints_reached,
    reached_goal,
    min_manhattan_distance,
)
```

## 5. RL và MDP trong repo

Một bài toán Reinforcement Learning thường được mô tả bằng MDP:

```text
MDP = (S, A, P, R, gamma)
```

| Ký hiệu | Trong repo này |
| --- | --- |
| `S` - State | Robot đang ở đâu, nhìn hướng nào, đã qua checkpoint nào, xung quanh có gì |
| `A` - Action | Tiến, lùi, quay trái, quay phải |
| `P` - Transition | Kết quả sau khi thực hiện action |
| `R` - Reward | Tín hiệu tốt/xấu sau mỗi action |
| `gamma` | Mức quan tâm tới reward trong tương lai |

### Episode

Một episode là một lần robot bắt đầu lại từ `start` và thử đi tới goal. Mỗi
episode có giới hạn số bước để tránh vòng lặp vô hạn.

### Policy

Policy là quy tắc chọn action từ state. Trong Q-learning, policy thường chọn
action có Q-value lớn nhất.

### Q-table

Q-table lưu giá trị:

```text
Q(state, action)
```

Q-value càng lớn nghĩa là action đó được dự đoán càng tốt khi robot đang ở
state tương ứng.

## 6. State: robot cần nhớ điều gì?

State được biểu diễn bằng class `RLState` trong `rl_library.py`. Class này có
thể chứa đầy đủ các thuộc tính dưới đây.

Mặc định, `Algo/StudentRL.py` dùng state tối giản trong `build_state()`:

```text
(x, y, dx, dy, checkpoints_mask, context)
```

Các trường cảm biến, hướng mục tiêu và recovery nhận giá trị mặc định. Đây là
điểm bắt đầu dễ hiểu; học sinh có thể thêm từng trường vào state và so sánh kết
quả.

| Thuộc tính | Ý nghĩa |
| --- | --- |
| `x`, `y` | Tọa độ hiện tại |
| `dx`, `dy` | Hướng nhìn hiện tại |
| `checkpoints_mask` | Các checkpoint đã đi qua |
| `context` | Tên/ngữ cảnh map để không trộn nhầm state giữa các map |
| `obstacle_mask` | Vật cản ở trước, phải, sau, trái so với robot |
| `target_forward` | Mục tiêu ở phía trước (`1`), sau (`-1`) hay ngang (`0`) |
| `target_right` | Mục tiêu ở bên phải (`1`), trái (`-1`) hay thẳng hàng (`0`) |
| `target_is_checkpoint` | Mục tiêu hiện tại có phải checkpoint không |
| `last_action` | Action vừa làm |
| `recovery_mode` | Robot vừa va vật cản và đang cần thoát ra |

Hướng robot dùng vector:

| Hằng số | Vector |
| --- | --- |
| `UP` | `(0, 1)` |
| `RIGHT` | `(1, 0)` |
| `DOWN` | `(0, -1)` |
| `LEFT` | `(-1, 0)` |

### Checkpoint mask

`checkpoints_mask` là một số nguyên dùng các bit để ghi nhớ checkpoint.

Ví dụ có 3 checkpoint:

```text
000: chưa qua checkpoint nào
001: đã qua checkpoint 0
101: đã qua checkpoint 0 và 2
111: đã qua cả 3 checkpoint
```

Không cần tự xử lý bit trong bài cơ bản. Có thể dùng:

```python
checkpoint_count(state.checkpoints_mask)
remaining_checkpoints(state, checkpoints)
```

### Obstacle mask

`obstacle_mask` cũng dùng bit:

| Hằng số | Giá trị |
| --- | ---: |
| `FRONT_BLOCKED` | `1` |
| `RIGHT_BLOCKED` | `2` |
| `BACK_BLOCKED` | `4` |
| `LEFT_BLOCKED` | `8` |

Ví dụ kiểm tra phía trước:

```python
if state.obstacle_mask & FRONT_BLOCKED:
    print("Phía trước bị chặn")
```

Đây là hướng **tương đối với robot**, không phải hướng cố định trên bản đồ.
Nếu robot quay, ý nghĩa trước/phải/sau/trái cũng quay theo.

### Exact state và micro-pattern state

Hàm `update_q_values()` của thư viện hỗ trợ học song song hai cách nhìn:

- **Exact state**: nhớ chính xác vị trí, hướng, map và checkpoint.
- **Micro-pattern state**: chỉ nhớ dạng tình huống cục bộ như “phía trước bị
  chặn, mục tiêu ở bên phải”.

Nhờ micro-pattern, kinh nghiệm tránh một vật cản có thể được dùng lại ở vị trí
khác hoặc map khác có tình huống tương tự. Micro-pattern sẽ có nhiều ý nghĩa
hơn sau khi `build_state()` giữ lại các trường như `obstacle_mask`,
`target_forward`, `target_right`, `last_action` và `recovery_mode`.

Tạo state đúng cách bằng hàm thư viện:

```python
state = get_state(
    position=start,
    direction=UP,
    checkpoints=checkpoints,
    map_obj=map_obj,
    goal=goal,
)
```

## 7. Action: robot được phép làm gì?

Repo có 4 action:

```python
FORWARD = "forward"
BACKWARD = "backward"
TURN_LEFT = "turn_left"
TURN_RIGHT = "turn_right"
```

Danh sách đầy đủ nằm trong:

```python
ACTIONS
```

Ý nghĩa:

- `FORWARD`: đi 1 ô theo hướng đang nhìn.
- `BACKWARD`: lùi 1 ô nhưng không đổi hướng nhìn.
- `TURN_LEFT`: quay trái 90 độ, chưa di chuyển.
- `TURN_RIGHT`: quay phải 90 độ, chưa di chuyển.

`valid_actions(...)` hiện trả về cả 4 action. Action đâm vào vật cản vẫn được
giữ lại để robot có thể thử, nhận reward âm và học rằng action đó không tốt.

## 8. Transition: action làm state thay đổi thế nào?

Dùng:

```python
result = transition(map_obj, state, action, goal, checkpoints)
```

`result` là một `TransitionResult` có các thông tin quan trọng:

| Thuộc tính | Ý nghĩa |
| --- | --- |
| `next_state` | State sau action |
| `reward` | Reward của action |
| `moved` | Có di chuyển thành công không |
| `turned` | Có quay không |
| `hit_obstacle` | Có đụng vật cản/biên map không |
| `reached_goal` | Có đang đứng ở goal không |
| `reached_checkpoints` | Số checkpoint đã qua |
| `detected_node` | Node vật cản vừa phát hiện |
| `detected_edge` | Cạnh vật cản vừa phát hiện |
| `message` | Mô tả bằng chữ |

Ví dụ:

```python
result = transition(map_obj, state, FORWARD, goal, checkpoints)
state = result.next_state
reward = result.reward
```

Episode hoàn thành khi:

```python
is_terminal_state(state, goal, checkpoints)
```

Hàm này chỉ trả về `True` khi robot đã qua tất cả checkpoint và đang ở goal.

## 9. Reward: robot học từ tín hiệu nào?

**Reward được tính sau từng action** và dùng để cập nhật Q-value. Reward không
phải score cuối.

Các thành phần reward:

| Sự kiện | Reward |
| --- | ---: |
| Di chuyển thành công | `-4` |
| Quay | `-1` |
| Đụng vật cản hoặc ra ngoài map | `-25` |
| Mỗi ô tiến gần mục tiêu hiện tại, khi mục tiêu không đổi | `+5` |
| Mỗi ô đi xa mục tiêu hiện tại, khi mục tiêu không đổi | `-5` |
| Lần đầu tới một checkpoint | `+200` |
| Hoàn thành tất cả checkpoint và tới goal | `+400` |
| Di chuyển thoát khỏi recovery mode | `+8` |
| Quay khi phía trước bị chặn | `+2` |
| Đụng vật cản lần nữa trong recovery mode | `-8` thêm |
| Vừa quay trái lại quay phải, hoặc ngược lại | `-2` thêm |

Mục tiêu hiện tại là checkpoint chưa đạt gần nhất. Khi hết checkpoint, mục tiêu
hiện tại mới là goal.

Một vài ví dụ:

- Đi 1 ô gần mục tiêu: `-4 + 5 = +1`.
- Đi 1 ô xa mục tiêu: `-4 - 5 = -9`.
- Đụng vật cản lần đầu: `-25`.
- Lặp lại va chạm khi đang recovery: `-25 - 8 = -33`.
- Tới goal hợp lệ khi bước đó tiến gần mục tiêu: `-4 + 5 + 400 = +401`.

Khi vừa tới checkpoint, mục tiêu hiện tại có thể đổi sang checkpoint tiếp theo
hoặc goal. Vì vậy reward chính xác của bước đó là `+200` cộng với các thành
phần di chuyển và khoảng cách tới mục tiêu mới, không phải lúc nào cũng là
`+201`.

Các hằng reward nằm ở đầu `rl_library.py`. Hàm tính reward chính là:

```python
transition_reward(result, goal, checkpoints)
```

Không nên cộng reward lần thứ hai sau khi gọi `transition()`, vì
`result.reward` đã được tính sẵn.

## 10. Q-learning trong `StudentRL.py`

Công thức cập nhật:

```text
Q(s,a) = Q(s,a)
       + alpha * [reward + gamma * max Q(s',a') - Q(s,a)]
```

Trong đó:

- `s`: state hiện tại.
- `a`: action vừa chọn.
- `s'`: state tiếp theo.
- `alpha`: learning rate, mức độ học từ trải nghiệm mới.
- `gamma`: mức quan tâm tới reward tương lai.

Code trong repo tách công thức thành:

```python
next_best_q = max_learned_q(qtable, result.next_state)
target = reward + gamma * next_best_q
update_q_values(qtable, state, action, target, alpha)
```

`update_q_values()` cập nhật cả exact state và micro-pattern state.

### Epsilon-greedy

Khi train, robot cần cân bằng:

- **Explore**: thử action ngẫu nhiên để tìm cách mới.
- **Exploit**: chọn action có Q-value cao nhất.

Quy tắc trong `choose_action()`:

```python
if rng.random() < epsilon:
    action = rng.choice(valid_actions(state, map_obj))
else:
    action = greedy_action(qtable, state, map_obj, goal, checkpoints)
```

Các tham số mặc định:

| Tham số | Mặc định | Ý nghĩa |
| --- | ---: | --- |
| `alpha` | `0.2` | Tốc độ học |
| `gamma` | `0.9` | Mức coi trọng tương lai |
| `epsilon_start` | `1.0` | Ban đầu khám phá nhiều |
| `epsilon_end` | `0.05` | Vẫn giữ ít nhất 5% khám phá |
| `epsilon_decay` | `0.995` | Giảm epsilon sau mỗi episode |

## 11. Các hàm thư viện nên dùng

Import từ `rl_library.py`:

```python
from rl_library import (
    ACTIONS,
    FORWARD,
    BACKWARD,
    TURN_LEFT,
    TURN_RIGHT,
    UP,
    FRONT_BLOCKED,
    get_state,
    valid_actions,
    transition,
    is_terminal_state,
    remaining_checkpoints,
    max_learned_q,
    update_q_values,
    greedy_action,
    checkpoint_count,
    calculate_score,
    manhattan,
)
```

### Nhóm state và checkpoint

| Hàm | Công dụng |
| --- | --- |
| `get_state(...)` | Tạo `RLState` đầy đủ từ vị trí, hướng, map và mục tiêu |
| `checkpoint_count(mask)` | Đếm số checkpoint đã đạt |
| `remaining_checkpoints(state, checkpoints)` | Lấy các checkpoint chưa đạt |
| `all_checkpoints_reached(state, checkpoints)` | Kiểm tra đã đủ checkpoint |
| `current_target(...)` | Chọn checkpoint gần nhất hoặc goal |
| `target_distance(...)` | Khoảng cách Manhattan tới mục tiêu hiện tại |
| `local_obstacle_mask(...)` | Đọc vật cản trước/phải/sau/trái |
| `micro_pattern_state(state)` | Chuyển state sang dạng tình huống dùng lại |

### Nhóm action và chuyển trạng thái

| Hàm | Công dụng |
| --- | --- |
| `valid_actions(state, map_obj)` | Trả về các action có thể chọn |
| `transition(...)` | Mô phỏng 1 action và trả về `TransitionResult` |
| `is_terminal_state(...)` | Kiểm tra episode đã hoàn thành |
| `turn_left(direction)` | Tính vector sau khi quay trái |
| `turn_right(direction)` | Tính vector sau khi quay phải |
| `direction_name(direction)` | Đổi vector hướng thành tên dễ đọc |

### Nhóm Q-table và policy

| Hàm | Công dụng |
| --- | --- |
| `greedy_action(...)` | Chọn action có Q-value tốt nhất |
| `max_learned_q(qtable, state)` | Lấy Q-value lớn nhất ở state |
| `update_q_values(...)` | Cập nhật Q-value exact và micro-pattern |
| `find_action_values(qtable, state)` | Tìm các Q-value phù hợp với state |
| `has_learned_action(qtable, state)` | Kiểm tra state đã có kinh nghiệm học |
| `action_visit_key(state, action)` | Tạo khóa đếm số lần lặp state-action |

### Nhóm reward và score

| Hàm | Công dụng |
| --- | --- |
| `transition_reward(...)` | Tính reward của một transition |
| `calculate_score(...)` | Tính score kết quả |
| `manhattan(a, b)` | Tính khoảng cách Manhattan |
| `min_manhattan(path, goal)` | Khoảng cách gần goal nhất trên một path |

### Nhóm map và file

| Hàm | Công dụng |
| --- | --- |
| `load_map_json(path)` | Đọc map JSON thành `MapData` |
| `save_map_json(...)` | Lưu map thành JSON |
| `map_dimensions(map_obj)` | Lấy `(width, height)` |
| `is_blocked_move(...)` | Kiểm tra một hướng đi có bị chặn |
| `shortest_known_path(...)` | Tìm đường ngắn trên phần map đã biết |

### Nhóm chạy thực tế nâng cao

| Hàm | Công dụng |
| --- | --- |
| `planned_action(...)` | Chọn action theo đường ngắn trên `known_map` |
| `exploration_action(...)` | Chọn action thoát kẹt/ít lặp lại |
| `runtime_action(...)` | Kết hợp policy, planner fallback và recovery |

Trong bài cơ bản, nên bắt đầu với `get_state`, `valid_actions`, `transition`,
`is_terminal_state`, `max_learned_q`, `update_q_values` và `greedy_action`.

## 12. Các hàm chính trong `Algo/StudentRL.py`

Phần đầu file có tiêu đề `KHU VỰC HỌC SINH`. Các TODO đều đã có cách làm mặc
định nên chương trình vẫn chạy trước khi học sinh sửa code.

Ba thành phần mặc định:

| Thành phần | Mặc định ban đầu |
| --- | --- |
| State | Vị trí, hướng, checkpoint và context map |
| Action | Cả 4 action: tiến, lùi, quay trái, quay phải |
| Reward | Dùng `result.reward` do `rl_library.transition()` tính |

| Hàm | Công dụng |
| --- | --- |
| `create_qtable()` | Tạo Q-table rỗng |
| `build_state(state)` | Chọn các thuộc tính được dùng làm khóa Q-table |
| `available_actions(...)` | Chọn tập action robot được phép thử |
| `best_action(...)` | Chọn action tốt nhất |
| `choose_action(...)` | Chọn action theo epsilon-greedy |
| `reward_function(...)` | Trả về reward dùng để học |
| `update_qtable(...)` | Cập nhật Q-value bằng công thức Q-learning |
| `run_episode(...)` | Chạy một episode, có thể học hoặc chỉ mô phỏng |
| `train(...)` | Chạy nhiều episode và trả về `TrainingResult` |
| `simulate_policy(...)` | Chạy policy với `epsilon = 0`, không cập nhật Q-table |
| `get_policy_path(...)` | Lấy danh sách tọa độ của policy |
| `save_policy_json(...)` | Lưu Q-table và kết quả |
| `load_policy_json(...)` | Đọc Q-table đã lưu |
| `format_episode_log(...)` | Tạo một dòng log dễ đọc |

Các TODO được sắp từ dễ đến khó:

1. Thay đổi thông tin trong `build_state()`.
2. Thay đổi tập action trong `available_actions()`.
3. Tự thiết kế reward trong `reward_function()`.
4. Tự viết cách chọn action trong `best_action()` và `choose_action()`.
5. Tự viết công thức cập nhật trong `update_qtable()`.

Luồng chính của một episode:

```text
Tạo state ban đầu
    |
Kiểm tra terminal
    |
Chọn action
    |
Gọi transition()
    |
Nhận next_state và reward
    |
Cập nhật Q-table
    |
Chuyển sang next_state và lặp lại
```

## 13. Policy fallback và recovery

Khi bấm `Run Student RL`, robot chạy trên `known_map`.

- Nếu Q-table đã biết state: dùng action từ policy.
- Nếu state còn lạ: dùng planner fallback để tìm action hợp lý.
- Nếu action bị chặn hoặc lặp lại: chuyển sang recovery/exploration.

Log có thể hiện một trong ba nguồn action:

- `policy`: action do Q-table chọn.
- `planner`: action do đường đi trên `known_map` gợi ý.
- `recovery`: action dùng để thoát vật cản hoặc vòng lặp.

Fallback giúp robot không đứng im ở state chưa từng gặp. Tuy nhiên mục tiêu của
bài học vẫn là cải thiện Q-table để tỷ lệ action từ `policy` tăng lên.

## 14. Chạy bằng dòng lệnh

Train một map:

```powershell
python -m tools.train_student_rl `
  --map .\maps\training\map_01_empty_grid.json `
  --episodes 500
```

Train lần lượt tất cả training map:

```powershell
python -m tools.train_student_rl --episodes 500
```

Chạy demo A*, Q-learning cơ bản và robot hai bản đồ:

```powershell
python -m tools.run_algorithm_demo
```

In một map mẫu ra terminal:

```powershell
python -m tools.visualize_map
```

Tạo lại bộ training map:

```powershell
python -m tools.generate_training_maps
```

## 15. Kiểm thử

Chạy toàn bộ test từ thư mục gốc:

```powershell
python -m unittest discover -s tests -v
```

Danh sách test được mô tả tại `tests/TEST_CASES.md`.

Sau khi sửa state, reward, transition hoặc Q-learning, nên chạy test để kiểm
tra các hành vi cũ có còn đúng không.

## 16. Cấu trúc repo

```text
Algo/
  StudentRL.py       # Thuật toán RL chính dành cho học sinh
  QLearning.py       # Q-learning cơ bản để tham khảo
  Astar.py           # A* để so sánh
app/
  UI.py              # Giao diện Tkinter
  services.py        # Kết nối UI, robot, map và thuật toán
src/
  map.py             # Node, Edge, LineMap
  agent.py           # Robot và known_map
maps/
  training/          # 30 map học theo độ khó tăng dần
  custom/            # Map tự tạo
  student_rl_policies/
                       # Policy Student RL đã lưu
tests/                # Automated tests
tools/                # Script train, demo, tạo và xem map
rl_library.py         # State, action, transition, reward và helper
main.py               # Điểm chạy chính của chương trình
```

## 17. Gợi ý khi thử nghiệm

Mỗi lần chỉ nên thay đổi một ý tưởng, ví dụ:

1. Thay reward khi tiến gần mục tiêu.
2. Thay tốc độ giảm epsilon.
3. Thêm hoặc bỏ một phần thông tin trong state.
4. So sánh số episode cần để học được đường hoàn chỉnh.

Ghi lại ít nhất các số sau trước và sau khi sửa:

- Có tới goal không?
- Đạt bao nhiêu checkpoint?
- Score bằng bao nhiêu?
- Tổng reward bằng bao nhiêu?
- Dùng bao nhiêu bước và bao nhiêu lần quay?

Một state quá ít thông tin làm robot không phân biệt được các tình huống. Một
state quá nhiều thông tin làm Q-table rất lớn và cần nhiều episode hơn. Đây là
một trong những bài toán quan trọng nhất khi thiết kế RL.
