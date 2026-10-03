"""GPU粒子系统 - 存在bug"""
from dataclasses import dataclass, field
from typing import Optional
import math


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


class ParticleSystem:
    def __init__(self, max_particles: int = 10000):
        self.max_particles = max_particles
        self.particles: list[Optional[Particle]] = [None] * max_particles
        self._next_id = 0
        self._free_indices: list[int] = []
        self.gravity: tuple[float, float, float] = (0, -9.8, 0)
        self.wind: tuple[float, float, float] = (0, 0, 0)
        self.emit_rate: float = 100.0  # 每秒发射数
        self._emit_accumulator: float = 0.0

    def emit_particles(self, count: int, position: tuple[float, float, float]) -> int:
        """发射粒子 - BUG: 只做位置初始化不做速度和加速度"""
        emitted = 0
        for i in range(count):
            if len(self._free_indices) > 0:
                idx = self._free_indices.pop()
            else:
                if self._next_id >= self.max_particles:
                    break
                idx = self._next_id
                self._next_id += 1
            # BUG: 速度和加速度都是0，粒子静止
            self.particles[idx] = Particle(
                id=idx,
                position=position,
                velocity=(0, 0, 0),  # BUG: 不随机速度
                acceleration=(0, 0, 0),  # BUG: 不应用重力
                life=5.0,
                max_life=5.0,
                size=1.0,
                color=(1, 1, 1, 1),
            )
            emitted += 1
        return emitted

    def update_particles(self, dt: float):
        """更新粒子 - BUG: 只做位置累加不做力场和碰撞"""
        for i, particle in enumerate(self.particles):
            if particle is None or not particle.active:
                continue
            # BUG: 不应用重力和风力
            # BUG: 不做碰撞检测
            particle.position = (
                particle.position[0] + particle.velocity[0] * dt,
                particle.position[1] + particle.velocity[1] * dt,
                particle.position[2] + particle.velocity[2] * dt,
            )
            particle.life -= dt
            if particle.life <= 0:
                # BUG: 不回收，只是标记不活跃
                particle.active = False

    def render_particles(self) -> list[dict]:
        """渲染粒子 - BUG: 只做点精灵不做广告牌"""
        result = []
        for particle in self.particles:
            if particle is None or not particle.active:
                continue
            # BUG: 不做广告牌，不做纹理动画
            result.append({
                "position": particle.position,
                "size": particle.size,
                "color": particle.color,
            })
        return result

    def sort_particles(self, camera_position: tuple[float, float, float]):
        """排序粒子 - BUG: 只按生成时间不按距离"""
        # BUG: 不实现排序
        pass

    def get_active_count(self) -> int:
        """获取活跃粒子数"""
        return sum(1 for p in self.particles if p is not None and p.active)

    def recycle_particle(self, index: int):
        """回收粒子 - BUG: 不实现"""
        # BUG: 不实现对象池回收
        pass
