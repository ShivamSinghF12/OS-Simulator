# OS-Simulator
A Simple OS simulator with basic functions like scheduling algorithms, file system and memory management, creation and deletion of files and much more
This project is OSSim (Operating System Simulator) — a rich, GUI-based educational tool built using Pygame to simulate the core components of an operating system. Its goal is to provide hands-on understanding of scheduling, memory management, storage systems, and deadlock handling.

!!!MAKE SURE TO INSTALL PYGAME USING pip install pygame in the terminal.

🔧 Core Features
🧠 Process Management
Process Creation: CPU-bound and IO-bound processes with randomized burst times and priorities.

Scheduling Algorithms:

Round Robin (RR)

First-Come-First-Serve (FCFS)

Shortest Job First (SJF)

Priority Scheduling

IO Simulation: IO-bound processes randomly request IO, causing temporary blocking.

Deadlock Simulation: Toggleable feature where processes can request limited resources (e.g., Printer, Disk) and may enter deadlock.

💾 Memory Management
Memory Allocation Algorithms:

First-Fit

Best-Fit

Worst-Fit

Visualization: Color-coded blocks showing free/used memory and process state.

Memory Compaction: Defragments memory space.

Fragmentation Calculation: Shown as a percentage.

📁 File & Storage System
File System with Directories

Create/delete/rename files and directories.

Navigate folders (cd, .. behavior included).

File Types: Text, logs, images, audio (simulated content).

Storage Allocation Algorithms: Same as memory.

Storage Compaction & Fragmentation Info

Search Files: Simple search functionality.

File Creation Sounds: Add realism.

🛑 Deadlock Detection & Resolution
Resource Manager:

Tracks which process holds or waits for resources.

Deadlock Detection:

Implements a Wait-For Graph algorithm to find cycles.

GUI shows which processes are involved.

Resolution:

Automatically selects the process with the lowest priority to terminate.

Frees up memory and resources, unblocks others.

📊 Performance Visualization
Gantt Chart Support (via execution history)

CPU Utilization Graph

Avg Turnaround and Waiting Time tracking.

🧰 UI & Controls
Button-driven interface (add process, compact memory/storage, save/load state, etc.)

Tooltips on hover

Text input for renaming files or searching

Dropdown for FPS control

Sound toggles for key events

💾 Save/Load Functionality
Pickle-based Save/Load of the full simulation state.

📋 Logging
All major actions are logged with timestamps in a file for analysis.


