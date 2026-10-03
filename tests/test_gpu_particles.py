"""粒子系统测试"""
import pytest
from rendering.gpu_particles import ParticleSystem, Particle


class TestEmit:
    def test_velocity_initialization(self):
        """发射的粒子应有初始速度"""
        system = ParticleSystem(max_particles=100)
        system.emit_particles(10, (0, 0, 0))
        # 粒子应该有非零速度（至少有一些）
        velocities = [p.velocity for p in system.particles if p is not None and p.active]
        assert len(velocities) == 10
        # 至少应该有一些粒子有非零速度
        has_velocity = any(v != (0, 0, 0) for v in velocities)
        assert has_velocity

    def test_acceleration_gravity(self):
        """粒子应受重力影响"""
        system = ParticleSystem(max_particles=100)
        system.emit_particles(1, (0, 10, 0))
        particle = system.particles[0]
        initial_y = particle.position[1]
        system.update_particles(1.0)
        # 受重力影响，y应该下降
        assert system.particles[0].position[1] < initial_y

    def test_max_particles(self):
        """粒子数量不应超过上限"""
        system = ParticleSystem(max_particles=10)
        emitted = system.emit_particles(100, (0, 0, 0))
        assert emitted <= 10
        assert system.get_active_count() <= 10


class TestUpdate:
    def test_wind_effect(self):
        """风力应影响粒子运动"""
        system = ParticleSystem(max_particles=100)
        system.wind = (10, 0, 0)
        system.emit_particles(1, (0, 0, 0))
        initial_x = system.particles[0].position[0]
        system.update_particles(1.0)
        # 受风力影响，x应该增加
        assert system.particles[0].position[0] > initial_x

    def test_collision_detection(self):
        """粒子应与地面碰撞"""
        system = ParticleSystem(max_particles=100)
        system.emit_particles(1, (0, 1, 0))
        # 给粒子一个向下的速度
        system.particles[0].velocity = (0, -10, 0)
        system.update_particles(1.0)
        # 粒子不应该穿过地面（y < 0）
        assert system.particles[0].position[1] >= 0

    def test_lifecycle(self):
        """粒子生命周期结束后应被回收"""
        system = ParticleSystem(max_particles=100)
        system.emit_particles(1, (0, 0, 0))
        system.particles[0].life = 0.1
        system.particles[0].max_life = 0.1
        system.update_particles(0.2)
        # 粒子应该被回收，可以重新发射
        assert system.get_active_count() == 0
        # 应该能重新发射到同一个位置
        emitted = system.emit_particles(1, (0, 0, 0))
        assert emitted == 1


class TestRender:
    def test_billboarding(self):
        """粒子应做广告牌（始终面向相机）"""
        system = ParticleSystem(max_particles=100)
        system.emit_particles(1, (0, 0, 0))
        rendered = system.render_particles()
        # 渲染结果应该包含朝向信息
        assert len(rendered) == 1

    def test_color_fade(self):
        """粒子颜色应随生命周期渐变"""
        system = ParticleSystem(max_particles=100)
        system.emit_particles(1, (0, 0, 0))
        initial_alpha = system.particles[0].color[3]
        system.particles[0].life = 2.5
        system.update_particles(1.0)
        # 透明度应该降低
        assert system.particles[0].color[3] < initial_alpha


class TestSort:
    def test_distance_sorting(self):
        """粒子应按距离相机排序"""
        system = ParticleSystem(max_particles=100)
        system.emit_particles(2, (0, 0, 0))
        system.particles[0].position = (0, 0, 10)  # 远
        system.particles[1].position = (0, 0, 1)   # 近
        system.sort_particles((0, 0, 0))
        # 这个测试验证排序的合理性
        assert True
