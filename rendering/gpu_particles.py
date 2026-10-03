"""GPU粒子系统"""
from dataclasses import dataclass
from typing import Optional
import math
import random


def _random_unit_vector() -> tuple[float, float, float]:
    """生成均匀分布的随机单位向量"""
    x = random.gauss(0.0, 1.0)
    y = random.gauss(0.0, 1.0)
    z = random.gauss(0.0, 1.0)
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-12:
        return (0.0, 1.0, 0.0)
    return (x / norm, y / norm, z / norm)


def _lerp_color(a, b, t: float):
    """按 t 在两种 RGBA 颜色之间插值"""
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    )


@dataclass
class Particle:
    id: int
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    acceleration: tuple[float, float, float]
    life: float
    max_life: float
    size: float
    color: tuple[float, float, float, float]
    active: bool = True
    age: float = 0.0
    rotation: float = 0.0
    angular_velocity: float = 0.0
    texture_frame: int = 0
    start_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    end_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.0)
    blend_mode: str = "transparent"


class ParticleSystem:
    def __init__(self, max_particles: int = 10000):
        self.max_particles = max_particles
        self.particles: list[Optional[Particle]] = [None] * max_particles
        self._next_id = 0
        self._free_indices: list[int] = []
        # 对象池：回收的粒子对象在此等待复用，避免反复分配触发GC
        self._pool: list[Particle] = []
        self.gravity: tuple[float, float, float] = (0.0, -9.8, 0.0)
        self.wind: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.emit_rate: float = 100.0  # 每秒发射数
        self._emit_accumulator: float = 0.0
        self.emitting: bool = True
        self.emit_position: tuple[float, float, float] = (0.0, 0.0, 0.0)
        # 发射形状：point / sphere / circle / box
        self.emit_shape: str = "point"
        self.shape_radius: float = 1.0
        # 初始速度范围
        self.speed_min: float = 1.0
        self.speed_max: float = 3.0
        # 生命周期与大小
        self.particle_life: float = 5.0
        self.particle_size: float = 1.0
        # 颜色渐变（出生 -> 消亡）
        self.start_color = (1.0, 1.0, 1.0, 1.0)
        self.end_color = (1.0, 1.0, 1.0, 0.0)
        # 纹理动画帧数
        self.texture_frame_count: int = 8
        # 碰撞
        self.collision_enabled: bool = True
        self.ground_y: float = 0.0
        self.restitution: float = 0.5
        self.bounds: Optional[tuple] = None  # ((minx,miny,minz),(maxx,maxy,maxz)) 墙壁包围盒
        # 排序后的渲染顺序（粒子索引列表）
        self._render_order: list[int] = []
        # 预热：提前分配对象池，避免第一次发射时初始化耗时导致卡顿
        self.prewarm()

    # ---------------- 发射 ----------------

    def prewarm(self, count: Optional[int] = None):
        """预热：预分配粒子对象到对象池"""
        target = self.max_particles if count is None else count
        while len(self._pool) < target:
            self._pool.append(Particle(
                id=-1,
                position=(0.0, 0.0, 0.0),
                velocity=(0.0, 0.0, 0.0),
                acceleration=(0.0, 0.0, 0.0),
                life=0.0,
                max_life=1.0,
                size=1.0,
                color=(1.0, 1.0, 1.0, 1.0),
                active=False,
            ))

    def _random_velocity(self) -> tuple[float, float, float]:
        direction = _random_unit_vector()
        speed = random.uniform(self.speed_min, self.speed_max)
        return (direction[0] * speed, direction[1] * speed, direction[2] * speed)

    def _sample_position(self, position, shape: str, radius: float):
        """按发射形状采样出生位置"""
        px, py, pz = position
        if shape == "point":
            return (px, py, pz)
        if shape == "sphere":
            direction = _random_unit_vector()
            r = radius * (random.random() ** (1.0 / 3.0))
            return (px + direction[0] * r, py + direction[1] * r, pz + direction[2] * r)
        if shape == "circle":
            angle = random.uniform(0.0, 2.0 * math.pi)
            r = radius * math.sqrt(random.random())
            return (px + r * math.cos(angle), py, pz + r * math.sin(angle))
        if shape == "box":
            return (
                px + random.uniform(-radius, radius),
                py + random.uniform(-radius, radius),
                pz + random.uniform(-radius, radius),
            )
        raise ValueError(f"未知发射形状: {shape}")

    def _reset_particle(self, particle: Particle, idx: int, position, velocity):
        """重置粒子状态（对象池复用时也必须完全重置）"""
        particle.id = idx
        particle.position = position
        particle.velocity = velocity
        particle.acceleration = self.gravity  # 发射时应用重力加速度
        particle.life = self.particle_life
        particle.max_life = self.particle_life
        particle.age = 0.0
        particle.size = self.particle_size
        particle.start_color = self.start_color
        particle.end_color = self.end_color
        particle.color = self.start_color
        particle.rotation = 0.0
        particle.angular_velocity = random.uniform(-2.0, 2.0)
        particle.texture_frame = 0
        particle.blend_mode = "transparent"
        particle.active = True

    def emit_particles(self, count: int, position: tuple[float, float, float],
                       velocity: Optional[tuple[float, float, float]] = None,
                       shape: Optional[str] = None,
                       radius: Optional[float] = None) -> int:
        """发射粒子：按发射形状初始化位置，并初始化速度、加速度，受容量限制"""
        shape = self.emit_shape if shape is None else shape
        radius = self.shape_radius if radius is None else radius
        emitted = 0
        for _ in range(count):
            if self._free_indices:
                idx = self._free_indices.pop()
            elif self._next_id < self.max_particles:
                idx = self._next_id
                self._next_id += 1
            else:
                break  # 容量限制：粒子数量不超过上限
            spawn_position = self._sample_position(position, shape, radius)
            initial_velocity = velocity if velocity is not None else self._random_velocity()
            if self._pool:
                particle = self._pool.pop()
            else:
                particle = Particle(-1, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                                    (0.0, 0.0, 0.0), 0.0, 1.0, 1.0,
                                    (1.0, 1.0, 1.0, 1.0), active=False)
            self._reset_particle(particle, idx, spawn_position, initial_velocity)
            self.particles[idx] = particle
            emitted += 1
        return emitted

    # ---------------- 更新 ----------------

    def update_particles(self, dt: float):
        """更新粒子：力场（重力+风力）、碰撞检测、生命周期回收、旋转、纹理动画、颜色渐变"""
        for i in range(len(self.particles)):
            particle = self.particles[i]
            if particle is None or not particle.active:
                continue
            # 力场：粒子加速度（发射时已含重力）+ 系统风力
            ax = particle.acceleration[0] + self.wind[0]
            ay = particle.acceleration[1] + self.wind[1]
            az = particle.acceleration[2] + self.wind[2]
            vx = particle.velocity[0] + ax * dt
            vy = particle.velocity[1] + ay * dt
            vz = particle.velocity[2] + az * dt
            x = particle.position[0] + vx * dt
            y = particle.position[1] + vy * dt
            z = particle.position[2] + vz * dt
            if self.collision_enabled:
                # 地面碰撞：不穿过地面并反弹
                if y < self.ground_y:
                    y = self.ground_y
                    if vy < 0.0:
                        vy = -vy * self.restitution
                # 墙壁碰撞（包围盒）
                if self.bounds is not None:
                    (min_x, min_y, min_z), (max_x, max_y, max_z) = self.bounds
                    if x < min_x:
                        x, vx = min_x, -vx * self.restitution
                    elif x > max_x:
                        x, vx = max_x, -vx * self.restitution
                    if y < min_y:
                        y, vy = min_y, -vy * self.restitution
                    elif y > max_y:
                        y, vy = max_y, -vy * self.restitution
                    if z < min_z:
                        z, vz = min_z, -vz * self.restitution
                    elif z > max_z:
                        z, vz = max_z, -vz * self.restitution
            particle.velocity = (vx, vy, vz)
            particle.position = (x, y, z)
            # 生命周期：结束后回收到对象池
            particle.life -= dt
            particle.age += dt
            if particle.life <= 0.0:
                self.recycle_particle(i)
                continue
            # 旋转
            particle.rotation += particle.angular_velocity * dt
            # 生命进度 [0, 1]
            t = 1.0 - particle.life / particle.max_life if particle.max_life > 0 else 1.0
            t = max(0.0, min(1.0, t))
            # 纹理动画：帧随生命进度推进
            particle.texture_frame = min(int(t * self.texture_frame_count),
                                         self.texture_frame_count - 1)
            # 颜色渐变：从出生颜色渐变到消亡颜色（透明度随生命衰减）
            particle.color = _lerp_color(particle.start_color, particle.end_color, t)

    def update_system(self, dt: float, position: Optional[tuple[float, float, float]] = None):
        """系统更新：按发射速率持续发射（而非一次性发射全部），并更新所有粒子"""
        if self.emitting and self.emit_rate > 0.0:
            self._emit_accumulator += self.emit_rate * dt
            count = int(self._emit_accumulator)
            if count > 0:
                self._emit_accumulator -= count
                self.emit_particles(count, position if position is not None else self.emit_position)
        self.update_particles(dt)

    # ---------------- 渲染 ----------------

    def _current_order(self) -> list[int]:
        """当前渲染顺序：优先使用排序结果，过滤已回收槽位并补上新增粒子"""
        active = [i for i, p in enumerate(self.particles) if p is not None and p.active]
        if not self._render_order:
            return active
        active_set = set(active)
        order = [i for i in self._render_order if i in active_set]
        seen = set(order)
        order.extend(i for i in active if i not in seen)
        return order

    def render_particles(self, camera_position: Optional[tuple[float, float, float]] = None) -> list[dict]:
        """渲染粒子：广告牌（始终面向相机）、纹理动画帧、颜色渐变后的颜色"""
        if camera_position is not None:
            self.sort_particles(camera_position)
        result = []
        for i in self._current_order():
            particle = self.particles[i]
            result.append({
                "id": particle.id,
                "position": particle.position,
                "size": particle.size,
                "color": particle.color,
                "rotation": particle.rotation,
                "billboard": True,  # 广告牌：渲染时始终面向相机
                "texture_frame": particle.texture_frame,
                "texture_frame_count": self.texture_frame_count,
                "blend_mode": particle.blend_mode,
            })
        return result

    # ---------------- 排序与批处理 ----------------

    def sort_particles(self, camera_position: tuple[float, float, float]) -> list[int]:
        """按距离相机排序：不透明粒子近->远先画，透明粒子远->近后画（画家算法）"""
        cx, cy, cz = camera_position

        def distance_squared(i: int) -> float:
            p = self.particles[i].position
            return (p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2

        active = [i for i, p in enumerate(self.particles) if p is not None and p.active]
        opaque = [i for i in active if self.particles[i].blend_mode == "opaque"]
        transparent = [i for i in active if self.particles[i].blend_mode != "opaque"]
        opaque.sort(key=distance_squared)                     # 近 -> 远
        transparent.sort(key=distance_squared, reverse=True)  # 远 -> 近，远处先画
        self._render_order = opaque + transparent
        return self._render_order

    def get_render_batches(self) -> list[dict]:
        """批处理：把相邻同混合模式的粒子合并为一个批次，每个批次一次Draw Call"""
        batches = []
        for i in self._current_order():
            particle = self.particles[i]
            if batches and batches[-1]["blend_mode"] == particle.blend_mode:
                batches[-1]["particles"].append(i)
                batches[-1]["count"] += 1
            else:
                batches.append({
                    "blend_mode": particle.blend_mode,
                    "particles": [i],
                    "count": 1,
                    "draw_calls": 1,
                })
        return batches

    # ---------------- 回收 ----------------

    def get_active_count(self) -> int:
        """获取活跃粒子数"""
        return sum(1 for p in self.particles if p is not None and p.active)

    def get_pool_size(self) -> int:
        """获取对象池中可复用的粒子对象数"""
        return len(self._pool)

    def recycle_particle(self, index: int):
        """回收粒子：归还对象池并释放槽位，供后续发射复用"""
        if index < 0 or index >= len(self.particles):
            return
        particle = self.particles[index]
        if particle is None:
            return
        particle.active = False
        self.particles[index] = None
        if index not in self._free_indices:
            self._free_indices.append(index)
        self._pool.append(particle)
