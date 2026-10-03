"""GPU粒子系统"""
from dataclasses import dataclass
from typing import Optional
import math
import random


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
    rotation: float = 0.0
    angular_velocity: float = 0.0
    transparent: bool = True
    active: bool = True


class ParticleSystem:
    def __init__(self, max_particles: int = 10000,
                 emission_shape: str = "point",
                 shape_radius: float = 1.0,
                 prewarm: bool = False):
        # 容量限制：粒子数量不可超过上限，防止内存无限增长
        self.max_particles = max_particles
        self.particles: list[Optional[Particle]] = [None] * max_particles
        self._next_id = 0
        # 对象池：回收的槽位复用，避免反复新建对象造成GC压力
        self._free_indices: list[int] = []
        self.gravity: tuple[float, float, float] = (0, -9.8, 0)
        self.wind: tuple[float, float, float] = (0, 0, 0)
        self.emit_rate: float = 100.0  # 每秒发射数
        self._emit_accumulator: float = 0.0
        self.emitter_position: tuple[float, float, float] = (0, 0, 0)
        # 发射形状：point（点发射）或 sphere（球发射）
        self.emission_shape = emission_shape
        self.shape_radius = shape_radius
        # 发射初速度范围
        self.min_speed: float = 0.5
        self.max_speed: float = 2.0
        # 颜色渐变：出生颜色 -> 消失颜色
        self.start_color: tuple[float, float, float, float] = (1, 1, 1, 1)
        self.end_color: tuple[float, float, float, float] = (1, 1, 1, 0)
        # 纹理动画帧数
        self.texture_frames: int = 8
        # 地面碰撞（y=0平面）与弹性系数
        self.ground_y: float = 0.0
        self.restitution: float = 0.4
        # 批处理大小（每个Draw Call提交的粒子数）
        self.batch_size: int = 256
        self._sorted_indices: list[int] = []
        if prewarm:
            self.prewarm()

    # ---------------- 发射 ----------------

    def _random_direction(self) -> tuple[float, float, float]:
        """球面上均匀随机方向"""
        z = random.uniform(-1.0, 1.0)
        theta = random.uniform(0.0, 2.0 * math.pi)
        r = math.sqrt(max(0.0, 1.0 - z * z))
        return (r * math.cos(theta), z, r * math.sin(theta))

    def _spawn_position(self, base: tuple[float, float, float]) -> tuple[float, float, float]:
        """按发射形状生成初始位置"""
        if self.emission_shape == "sphere":
            direction = self._random_direction()
            radius = self.shape_radius * random.random() ** (1.0 / 3.0)
            return (base[0] + direction[0] * radius,
                    base[1] + direction[1] * radius,
                    base[2] + direction[2] * radius)
        return base

    def _spawn_velocity(self) -> tuple[float, float, float]:
        """随机初速度"""
        direction = self._random_direction()
        speed = random.uniform(self.min_speed, self.max_speed)
        return (direction[0] * speed, direction[1] * speed, direction[2] * speed)

    def emit_particles(self, count: int, position: tuple[float, float, float]) -> int:
        """发射粒子：随机初速度、初始加速度、按发射形状分布，受容量限制"""
        emitted = 0
        for _ in range(count):
            if self._free_indices:
                idx = self._free_indices.pop()
            elif self._next_id < self.max_particles:
                idx = self._next_id
                self._next_id += 1
            else:
                break  # 容量已满
            self.particles[idx] = Particle(
                id=idx,
                position=self._spawn_position(position),
                velocity=self._spawn_velocity(),
                acceleration=(0.0, 0.0, 0.0),
                life=5.0,
                max_life=5.0,
                size=1.0,
                color=self.start_color,
                rotation=random.uniform(0.0, 2.0 * math.pi),
                angular_velocity=random.uniform(-1.0, 1.0),
            )
            emitted += 1
        return emitted

    # ---------------- 更新 ----------------

    def update_particles(self, dt: float):
        """更新粒子：力场（重力+风力）、碰撞检测、生命周期回收、颜色渐变"""
        for i, particle in enumerate(self.particles):
            if particle is None or not particle.active:
                continue
            # 力场：重力与风力叠加到速度
            ax = particle.acceleration[0] + self.gravity[0] + self.wind[0]
            ay = particle.acceleration[1] + self.gravity[1] + self.wind[1]
            az = particle.acceleration[2] + self.gravity[2] + self.wind[2]
            vx = particle.velocity[0] + ax * dt
            vy = particle.velocity[1] + ay * dt
            vz = particle.velocity[2] + az * dt
            px = particle.position[0] + vx * dt
            py = particle.position[1] + vy * dt
            pz = particle.position[2] + vz * dt
            # 碰撞检测：地面反弹，不允许穿透
            if py < self.ground_y:
                py = self.ground_y
                vy = -vy * self.restitution
                vx *= 0.8
                vz *= 0.8
            particle.velocity = (vx, vy, vz)
            particle.position = (px, py, pz)
            particle.rotation += particle.angular_velocity * dt
            # 生命周期
            particle.life -= dt
            if particle.life <= 0:
                self.recycle_particle(i)
                continue
            # 颜色渐变：按剩余生命在出生/消失颜色间插值
            t = 1.0 - particle.life / particle.max_life
            particle.color = tuple(
                self.start_color[c] + (self.end_color[c] - self.start_color[c]) * t
                for c in range(4)
            )

    # ---------------- 渲染 ----------------

    def render_particles(self,
                         camera_position: tuple[float, float, float] = (0, 0, -10),
                         camera_forward: tuple[float, float, float] = (0, 0, 1),
                         camera_up: tuple[float, float, float] = (0, 1, 0)) -> list[dict]:
        """渲染粒子：广告牌朝向相机、纹理动画帧、颜色渐变，按排序顺序输出"""
        # 广告牌基向量：right = normalize(cross(forward, up))
        fx, fy, fz = camera_forward
        ux, uy, uz = camera_up
        rx = fy * uz - fz * uy
        ry = fz * ux - fx * uz
        rz = fx * uy - fy * ux
        length = math.sqrt(rx * rx + ry * ry + rz * rz) or 1.0
        right = (rx / length, ry / length, rz / length)
        up = camera_up

        order = self._sorted_indices or [
            i for i, p in enumerate(self.particles) if p is not None and p.active
        ]
        result = []
        for i in order:
            particle = self.particles[i]
            if particle is None or not particle.active:
                continue
            half = particle.size / 2.0
            cos_r = math.cos(particle.rotation)
            sin_r = math.sin(particle.rotation)
            # 粒子自旋后的广告牌偏移
            corners = []
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                ox = (sx * cos_r - sy * sin_r) * half
                oy = (sx * sin_r + sy * cos_r) * half
                corners.append((
                    particle.position[0] + right[0] * ox + up[0] * oy,
                    particle.position[1] + right[1] * ox + up[1] * oy,
                    particle.position[2] + right[2] * ox + up[2] * oy,
                ))
            # 纹理动画：帧随生命周期推进
            life_t = 1.0 - particle.life / particle.max_life
            frame = min(int(life_t * self.texture_frames), self.texture_frames - 1)
            result.append({
                "position": particle.position,
                "size": particle.size,
                "color": particle.color,
                "billboard": True,
                "quad": corners,
                "rotation": particle.rotation,
                "texture_frame": frame,
                "transparent": particle.transparent,
            })
        return result

    # ---------------- 排序 ----------------

    def sort_particles(self, camera_position: tuple[float, float, float]) -> list[list[int]]:
        """按距离相机排序（远->近，透明粒子正确混合），并分批以降低Draw Call"""
        cx, cy, cz = camera_position
        active = [i for i, p in enumerate(self.particles) if p is not None and p.active]

        def distance_sq(i: int) -> float:
            p = self.particles[i].position
            return (p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2

        # 透明粒子：从远到近渲染；不透明粒子放最前
        opaque = sorted((i for i in active if not self.particles[i].transparent),
                        key=distance_sq)
        transparent = sorted((i for i in active if self.particles[i].transparent),
                             key=distance_sq, reverse=True)
        self._sorted_indices = opaque + transparent
        # 批处理：把排序后的粒子按batch_size分组，每组一次Draw Call
        return [self._sorted_indices[j:j + self.batch_size]
                for j in range(0, len(self._sorted_indices), self.batch_size)]

    # ---------------- 系统 ----------------

    def update_system(self, dt: float):
        """系统更新：按发射速率持续发射，再推进粒子模拟"""
        # 发射速率：按时间累积，避免一次性发射全部粒子造成卡顿
        self._emit_accumulator += self.emit_rate * dt
        count = int(self._emit_accumulator)
        if count > 0:
            self._emit_accumulator -= count
            self.emit_particles(count, self.emitter_position)
        self.update_particles(dt)

    def prewarm(self, duration: float = 2.0, dt: float = 1.0 / 60.0):
        """预热：提前模拟一段时间，避免首次发射时初始化卡顿"""
        steps = int(duration / dt)
        for _ in range(steps):
            self.update_system(dt)

    def get_active_count(self) -> int:
        """获取活跃粒子数"""
        return sum(1 for p in self.particles if p is not None and p.active)

    def recycle_particle(self, index: int):
        """回收粒子到对象池，槽位可复用"""
        if 0 <= index < len(self.particles) and self.particles[index] is not None:
            self.particles[index] = None
            self._free_indices.append(index)
