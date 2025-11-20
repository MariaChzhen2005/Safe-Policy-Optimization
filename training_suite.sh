# PPO
python safepo/single_agent/ppo.py --task SafetyPointPush2-v0 --device cuda --device-id 0 --experiment single_agent_exp --seed 0
python safepo/single_agent/ppo.py --task SafetyPointButton2-v0 --device cuda --device-id 0 --experiment single_agent_exp --seed 0
python safepo/single_agent/ppo.py --task SafetyPointGoal2-v0 --device cuda --device-id 0 --experiment single_agent_exp --seed 0

# TRPO
python safepo/single_agent/trpo.py --task SafetyPointPush2-v0 --device cuda --device-id 0 --experiment single_agent_exp --seed 0
python safepo/single_agent/trpo.py --task SafetyPointButton2-v0 --device cuda --device-id 0 --experiment single_agent_exp --seed 0
python safepo/single_agent/trpo.py --task SafetyPointGoal2-v0 --device cuda --device-id 0 --experiment single_agent_exp --seed 0

# PPO-Lag
python safepo/single_agent/ppo_lag.py --task SafetyPointPush2-v0 --device cuda --device-id 0 --experiment single_agent_exp --seed 0
python safepo/single_agent/ppo_lag.py --task SafetyPointButton2-v0 --device cuda --device-id 0 --experiment single_agent_exp --seed 0
python safepo/single_agent/ppo_lag.py --task SafetyPointGoal2-v0 --device cuda --device-id 0 --experiment single_agent_exp --seed 0