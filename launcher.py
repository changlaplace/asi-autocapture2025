
import subprocess
import time
import psutil
import os
from utlis import setup_logger

MAX_RESTARTS = 50

MEM_THRESHOLD = 80.0  # 80%


def launch_once():
    return subprocess.Popen(
        ["python", "capture_imgnet.py"],
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    )

def kill_process_tree(parent_pid, logger):
    try:
        parent = psutil.Process(parent_pid)
        children = parent.children(recursive=True)
        for child in children:
            logger.info(f"Terminating child process: PID={child.pid}")
            child.terminate()
        parent.terminate()
    except Exception as e:
        logger.error(f"Failed to terminate process tree: {e}")

if __name__ == '__main__':
    restart_count = 0
    logger = setup_logger(r'./launcher_logs')

    try:
        while restart_count < MAX_RESTARTS:
            logger.info(f"[launcher] Starting capture_imgnet.py (attempt {restart_count + 1})")
            p = launch_once()
            p_pid = p.pid

            while True:
                if p.poll() is not None:
                    break  # process has exited

                mem_percent = psutil.virtual_memory().percent
                logger.info(f"Memory usage {mem_percent:.2f}% currently")
                if mem_percent > MEM_THRESHOLD:
                    logger.warning(f"Memory usage {mem_percent:.2f}% exceeds {MEM_THRESHOLD}%. Killing process tree.")
                    kill_process_tree(p_pid, logger)
                    p.wait()
                    break

                time.sleep(60)

            logger.info(f"[launcher] Child exited with code {p.returncode}")
            restart_count += 1
            if restart_count >= MAX_RESTARTS:
                logger.info(f"[launcher] Reached max restart attempts ({MAX_RESTARTS}). Stopping.")
                break

            logger.info("[launcher] Restarting in 10s...\n")
            time.sleep(10)

    except KeyboardInterrupt:
        logger.info("[launcher] Ctrl+C received. Terminating if needed...")
        if 'p' in locals() and p.poll() is None:
            kill_process_tree(p.pid, logger)
        logger.info("[launcher] Exiting.")