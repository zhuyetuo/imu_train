"""Start phase 3 right away (don't wait for phase 2: its big SSL pretraining keeps the GPU idle while reading files).
File name ends with queue_run3.py on purpose: phase 4 waits for a process whose command ends with queue_run3.py."""
import queue_run3 as Q3

if __name__ == "__main__":
    Q3.Q2.main()
