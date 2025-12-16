@echo off
start "Worker1" python parallel_optune.py --n-trials 10
start "Worker2" python parallel_optune.py --n-trials 10
start "Worker3" python parallel_optune.py --n-trials 10
start "Worker4" python parallel_optune.py --n-trials 10
start "Worker5" python parallel_optune.py --n-trials 10
start "Worker6" python parallel_optune.py --n-trials 10
start "Worker7" python parallel_optune.py --n-trials 10
start "Worker8" python parallel_optune.py --n-trials 10