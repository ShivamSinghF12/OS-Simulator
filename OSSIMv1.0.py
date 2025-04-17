import pygame
import random
import os
import logging
import pickle
from datetime import datetime
import math

# Setup logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=f"ossim_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
logger = logging.getLogger("OSSim")

# Configuration settings

class Config:
    # Window settings
    WIDTH, HEIGHT = 1000, 850  # Increased height to accommodate taskbar
    FPS = 10
    FPS_OPTIONS = [1, 2, 5, 10, 15, 20]  # Available FPS options
    
    # Colors (changed them to a be bit softer on the eyes.)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (46, 204, 113)  # Softer green
    RED = (231, 76, 60)     # Softer red
    BLUE = (52, 152, 219)   # Softer blue
    LIGHT_BLUE = (174, 214, 241)  # Lighter, softer blue
    GRAY = (189, 195, 199)  # Softer gray
    DARK_GRAY = (127, 140, 141)   # Softer dark gray
    YELLOW = (241, 196, 15)  # Softer yellow
    ORANGE = (230, 126, 34)  # Softer orange
    PURPLE = (155, 89, 182)  # Softer purple
    TEAL = (26, 188, 156)    # New teal color
    DARK_BLUE = (41, 128, 185)  # New dark blue
    LIGHT_GREEN = (46, 204, 113)  # New light green
    LIGHT_PURPLE = (142, 68, 173)  # New light purple
    
    # Resource paths for sound files may add more in the future.
    SOUND_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Process settings
    MIN_BURST_TIME = 3
    MAX_BURST_TIME = 10
    MIN_PRIORITY = 1
    MAX_PRIORITY = 5
    
    # Memory settings
    MEMORY_SIZE = 1024
    
    # Storage settings
    STORAGE_SIZE = 1024
    MIN_FILE_SIZE = 50
    MAX_FILE_SIZE = 200
    
    # UI settings
    PROCESS_HEIGHT = 30
    MAX_DISPLAY_FILES = 3

# Helper functions
def load_sound(filename):
    try:
        sound_path = os.path.join(Config.SOUND_DIR, filename)
        if os.path.exists(sound_path) and pygame.mixer.get_init():
            return pygame.mixer.Sound(sound_path)
        return None
    except pygame.error as e:
        logger.error(f"Failed to load sound {filename}: {e}")
        return None

# Initialize Pygame and Mixer
def initialize_pygame():
    try:
        pygame.init()
        pygame.mixer.init()
        screen = pygame.display.set_mode((Config.WIDTH, Config.HEIGHT))
        pygame.display.set_caption("OSSim - Operating System Simulator")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 20)
        return screen, clock, font
    except Exception as e:
        logger.critical(f"Failed to initialize Pygame: {e}")
        raise

# Process Class 
class Process:
    def __init__(self, pid, arrival_time, burst_time, priority=1, io_bound=False):
        self.pid = pid
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.remaining_time = burst_time
        self.priority = priority
        self.state = "Ready"
        self.completion_time = None
        self.x, self.y = 20, 70
        self.io_bound = io_bound
        self.io_wait_time = 0
        self.io_frequency = random.randint(1, 3) if io_bound else 0
        self.color = (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255)
        )
        logger.info(f"Created Process PID:{pid} Burst:{burst_time} Priority:{priority} IO:{io_bound}")
    
    def __str__(self):
        return f"Process {self.pid}: {self.state}, Remaining: {self.remaining_time}"

# Scheduler Class with multiple algorithms
class Scheduler:
    def __init__(self, algorithm="RR", quantum=2):
        self.ready_queue = []
        self.algorithm = algorithm
        self.quantum = quantum
        self.current_time = 0
        self.current_process = None
        self.terminated = []
        self.io_queue = []
        self.time_slice = 0
        logger.info(f"Scheduler initialized with algorithm: {algorithm}, quantum: {quantum}")
    
    def add_process(self, process):
        self.ready_queue.append(process)
        logger.info(f"Added process {process.pid} to ready queue")
    
    def step(self):
        """Advance the simulation by one time unit."""
        try:
            previous_terminated_count = len(self.terminated)

            # Handle IO processes first
            for i in range(len(self.io_queue)-1, -1, -1):
                process = self.io_queue[i]
                process.io_wait_time -= 1
                if process.io_wait_time <= 0:
                    process.state = "Ready"
                    self.ready_queue.append(process)
                    self.io_queue.pop(i)
                    logger.info(f"Process {process.pid} returned from IO")
                    # Update Gantt chart for IO completion
                    if self.last_pid is not None:
                        self.execution_history.append((self.last_pid, self.last_start_time, self.current_time))
                    self.last_pid = process.pid
                    self.last_start_time = self.current_time

            # Select a process to run if none is running
            if not self.current_process:
                # Filter out blocked processes before selecting
                self.ready_queue = [p for p in self.ready_queue if p.state != "Blocked"]

                if not self.ready_queue:
                    return  # No ready process to run

                # Select process based on scheduling algorithm
                if self.algorithm == "FCFS":
                    self.current_process = self.ready_queue.pop(0)
                elif self.algorithm == "RR":
                    self.current_process = self.ready_queue.pop(0)
                    self.time_slice = 0
                elif self.algorithm == "SJF":
                    self.ready_queue.sort(key=lambda p: p.remaining_time)
                    self.current_process = self.ready_queue.pop(0)
                elif self.algorithm == "Priority":
                    self.ready_queue.sort(key=lambda p: p.priority, reverse=True)
                    self.current_process = self.ready_queue.pop(0)

                if self.current_process:
                    self.current_process.state = "Running"
                    logger.info(f"Selected process {self.current_process.pid} to run")
                    # Update Gantt chart for new process
                    if self.last_pid is not None:
                        self.execution_history.append((self.last_pid, self.last_start_time, self.current_time))
                    self.last_pid = self.current_process.pid
                    self.last_start_time = self.current_time

            # Process execution
            if self.current_process:
                if self.current_process.state == "Blocked":
                    return  # Don't run blocked processes

                # Update process execution time
                self.current_process.remaining_time -= 1
                self.current_time += 1
                self.time_slice += 1
                
                # Handle IO requests
                if (self.current_process.io_bound and 
                    random.random() < 0.2 and 
                    self.current_process.remaining_time > 0):
                    self.current_process.state = "IO"
                    self.current_process.io_wait_time = random.randint(1, 5)
                    self.io_queue.append(self.current_process)
                    logger.info(f"Process {self.current_process.pid} requested IO, waiting for {self.current_process.io_wait_time}")
                    # Update Gantt chart for IO start
                    if self.last_pid is not None:
                        self.execution_history.append((self.last_pid, self.last_start_time, self.current_time))
                    self.last_pid = None
                    self.current_process = None
                    return
                
                # Deadlock simulation logic
                if self.deadlock_simulation_enabled:
                    proc = self.current_process
                    if not hasattr(proc, "held_resources"):
                        proc.held_resources = []

                    # Only apply deadlock simulation to IO-bound processes
                    if proc.io_bound:
                        # If process is not holding any resources, it must acquire one first
                        if not proc.held_resources:
                            # Try to acquire any available resource
                            available_resources = ["Printer", "Disk", "Scanner"]
                            for resource in available_resources:
                                if self.resource_manager.request(proc.pid, resource):
                                    proc.held_resources.append(resource)
                                    logger.info(f"Process {proc.pid} acquired resource {resource}")
                                    return
                            
                            # If no resources available, process is blocked
                            proc.state = "Blocked"
                            self.scheduler.ready_queue.append(proc)
                            self.scheduler.current_process = None
                            self.blocked_processes.append(proc)
                            logger.info(f"Process {proc.pid} blocked - no resources available")
                            # Update Gantt chart for blocking
                            if self.last_pid is not None:
                                self.execution_history.append((self.last_pid, self.last_start_time, self.current_time))
                            self.last_pid = None
                            return
                        
                        # If process is holding a resource, it can request another one
                        elif random.random() < 0.3:  # 30% chance to request another resource
                            available_resources = ["Printer", "Disk", "Scanner"]
                            current_resource = proc.held_resources[0]
                            
                            # Try to request a different resource
                            for resource in available_resources:
                                if resource != current_resource:  # Don't request the same resource
                                    for other_pid, resources in self.resource_manager.allocated.items():
                                        if resource in resources and other_pid != proc.pid:
                                            if self.resource_manager.request(proc.pid, resource):
                                                proc.held_resources.append(resource)
                                                logger.info(f"Process {proc.pid} acquired additional resource {resource}")
                                            else:
                                                proc.state = "Blocked"
                                                self.scheduler.ready_queue.append(proc)
                                                self.scheduler.current_process = None
                                                self.blocked_processes.append(proc)
                                                logger.info(f"Process {proc.pid} blocked requesting {resource} while holding {current_resource}")
                                                # Update Gantt chart for blocking
                                                if self.last_pid is not None:
                                                    self.execution_history.append((self.last_pid, self.last_start_time, self.current_time))
                                                self.last_pid = None
                                                return
                        
                        # Process can release its resource
                        elif random.random() < 0.2:  # 20% chance to release the resource
                            resource = proc.held_resources[0]
                            self.resource_manager.release(proc.pid)
                            proc.held_resources.remove(resource)
                            logger.info(f"Process {proc.pid} released resource {resource}")

                # Check if process completed
                if self.current_process.remaining_time <= 0:
                    self.current_process.state = "Terminated"
                    self.current_process.completion_time = self.current_time
                    self.terminated.append(self.current_process)
                    
                    # Clean up memory and resources for completed process
                    if self.current_process in self.memory.allocated:
                        start, size = self.memory.allocated[self.current_process]
                        self.memory.blocks.append((start, size))
                        del self.memory.allocated[self.current_process]
                        self.memory._merge_adjacent_blocks()
                    
                    self.resource_manager.release(self.current_process.pid)
                    logger.info(f"Process {self.current_process.pid} terminated at time {self.current_time}")
                    # Update Gantt chart for termination
                    if self.last_pid is not None:
                        self.execution_history.append((self.last_pid, self.last_start_time, self.current_time))
                    self.last_pid = None
                    self.current_process = None
                    
                    # Play termination sound and update status
                    if self.terminate_sound:
                        self.terminate_sound.play()
                    self.status_message = f"Process {self.terminated[-1].pid} terminated"
                    self.update_performance_metrics()
                # Check if time quantum expired for Round Robin
                elif self.algorithm == "RR" and self.time_slice >= self.quantum:
                    self.current_process.state = "Ready"
                    self.ready_queue.append(self.current_process)
                    logger.info(f"Process {self.current_process.pid} time quantum expired, returning to ready queue")
                    # Update Gantt chart for quantum expiration
                    if self.last_pid is not None:
                        self.execution_history.append((self.last_pid, self.last_start_time, self.current_time))
                    self.last_pid = None
                    self.current_process = None
                    self.time_slice = 0

            # Track process execution for Gantt chart
            if self.current_process:
                current_pid = self.current_process.pid
                if self.last_pid != current_pid:
                    if self.last_pid is not None:
                        self.execution_history.append((self.last_pid, self.last_start_time, self.current_time))
                    self.last_pid = current_pid
                    self.last_start_time = self.current_time
            elif self.last_pid is not None:
                self.execution_history.append((self.last_pid, self.last_start_time, self.current_time))
                self.last_pid = None

            # Handle termination
            if len(self.terminated) > previous_terminated_count:
                terminated_proc = self.terminated[-1]
                self.memory.deallocate(terminated_proc)
                self.resource_manager.release(terminated_proc.pid)
                self.status_message = f"Process {terminated_proc.pid} terminated"

                if self.terminate_sound:
                    self.terminate_sound.play()

                self.update_performance_metrics()

            # Check for deadlock
            if self.deadlock_simulation_enabled:
                deadlock_detected, deadlock_info = self.resource_manager.detect_deadlock()
                if deadlock_detected:
                    status_parts = ["⚠️ Deadlock detected!"]
                    for cycle in deadlock_info:
                        for pid, resource in cycle:
                            # Get what the process is holding
                            holding = []
                            for held_pid, resources in self.resource_manager.allocated.items():
                                if held_pid == pid:
                                    holding.extend(resources)
                            
                            # Get what the process is waiting for
                            waiting_for = []
                            for req_pid, resources in self.resource_manager.requests.items():
                                if req_pid == pid:
                                    waiting_for.extend(resources)
                            
                            status_parts.append(f"PID {pid}: Holding {holding if holding else 'nothing'}, Waiting for {waiting_for if waiting_for else 'nothing'}")
                    self.status_message = "\n".join(status_parts)

        except Exception as e:
            logger.error(f"Error in step: {e}")
            self.status_message = f"Error: {str(e)}"

    def change_algorithm(self, algorithm):
        self.algorithm = algorithm
        # Reset time slice when changing algorithm
        self.time_slice = 0
        logger.info(f"Changed scheduling algorithm to {algorithm}")

# Memory Class with enhanced allocation algorithms
class Memory:
    def __init__(self, total_size=Config.MEMORY_SIZE):
        self.total_size = total_size
        self.blocks = [(0, total_size)]  # List of free blocks (start, size)
        self.allocated = {}  # Maps process to (start, size)
        self.algorithm = "First-Fit"  # Default allocation algorithm
        logger.info(f"Memory initialized with size {total_size}KB")
    
    def allocate(self, process, size):
        if self.algorithm == "First-Fit":
            return self._first_fit_allocate(process, size)
        elif self.algorithm == "Best-Fit":
            return self._best_fit_allocate(process, size)
        elif self.algorithm == "Worst-Fit":
            return self._worst_fit_allocate(process, size)
    
    def _first_fit_allocate(self, process, size):
        for i, (start, block_size) in enumerate(self.blocks):
            if block_size >= size:
                # Allocate memory from this block
                self.blocks[i] = (start + size, block_size - size)
                if self.blocks[i][1] == 0:
                    self.blocks.pop(i)
                self.allocated[process] = (start, size)
                logger.info(f"Allocated {size}KB at position {start} for process {process.pid} using First-Fit")
                return True
        logger.warning(f"Failed to allocate {size}KB for process {process.pid}: No suitable block found")
        return False
    
    def _best_fit_allocate(self, process, size):
        best_fit_index = -1
        best_fit_size = float('inf')
        
        for i, (start, block_size) in enumerate(self.blocks):
            if size <= block_size < best_fit_size:
                best_fit_index = i
                best_fit_size = block_size
        
        if best_fit_index != -1:
            start, block_size = self.blocks[best_fit_index]
            self.blocks[best_fit_index] = (start + size, block_size - size)
            if self.blocks[best_fit_index][1] == 0:
                self.blocks.pop(best_fit_index)
            self.allocated[process] = (start, size)
            logger.info(f"Allocated {size}KB at position {start} for process {process.pid} using Best-Fit")
            return True
        
        logger.warning(f"Failed to allocate {size}KB for process {process.pid}: No suitable block found")
        return False
    
    def _worst_fit_allocate(self, process, size):
        worst_fit_index = -1
        worst_fit_size = -1
        
        for i, (start, block_size) in enumerate(self.blocks):
            if size <= block_size and block_size > worst_fit_size:
                worst_fit_index = i
                worst_fit_size = block_size
        
        if worst_fit_index != -1:
            start, block_size = self.blocks[worst_fit_index]
            self.blocks[worst_fit_index] = (start + size, block_size - size)
            if self.blocks[worst_fit_index][1] == 0:
                self.blocks.pop(worst_fit_index)
            self.allocated[process] = (start, size)
            logger.info(f"Allocated {size}KB at position {start} for process {process.pid} using Worst-Fit")
            return True
        
        logger.warning(f"Failed to allocate {size}KB for process {process.pid}: No suitable block found")
        return False
    
    def deallocate(self, process):
        if process in self.allocated:
            start, size = self.allocated.pop(process)
            self.blocks.append((start, size))
            self._merge_adjacent_blocks()
            logger.info(f"Deallocated {size}KB at position {start} for process {process.pid}")
            return True
        
        logger.warning(f"Failed to deallocate memory for process {process.pid}: Not allocated")
        return False
    
    def _merge_adjacent_blocks(self):
        self.blocks.sort()  # Sort blocks by start position
        i = 0
        while i < len(self.blocks) - 1:
            current_start, current_size = self.blocks[i]
            next_start, next_size = self.blocks[i + 1]
            
            if current_start + current_size == next_start:
                # Merge adjacent blocks
                self.blocks[i] = (current_start, current_size + next_size)
                self.blocks.pop(i + 1)
            else:
                i += 1
    
    def compact(self):
        if not self.allocated:
            self.blocks = [(0, self.total_size)]
            logger.info("Memory compacted (no allocated blocks)")
            return
        
        # Sort allocated blocks by start position
        allocated_list = sorted(self.allocated.items(), key=lambda x: x[1][0])
        
        # Create new allocation
        new_allocated = {}
        next_position = 0
        
        for process, (_, size) in allocated_list:
            new_allocated[process] = (next_position, size)
            next_position += size
        
        # Update allocated blocks
        self.allocated = new_allocated
        
        # Update free blocks
        if next_position < self.total_size:
            self.blocks = [(next_position, self.total_size - next_position)]
        else:
            self.blocks = []
        
        logger.info("Memory compacted")
    
    def fragmentation(self):
        if not self.blocks:
            return 0
        
        total_free = sum(size for _, size in self.blocks)
        largest_free = max(size for _, size in self.blocks) if self.blocks else 0
        
        if total_free == 0:
            return 0
        
        return (total_free - largest_free) / self.total_size
    
    def change_algorithm(self, algorithm):
        self.algorithm = algorithm
        logger.info(f"Changed memory allocation algorithm to {algorithm}")

# Storage Class with enhanced features
class Storage:
    def __init__(self, total_size=Config.STORAGE_SIZE):
        self.total_size = total_size
        self.blocks = [(0, total_size)]  # List of free blocks (start, size)
        self.allocated = {}  # Maps filename to (start, size)
        self.algorithm = "First-Fit"  # Default allocation algorithm
        logger.info(f"Storage initialized with size {total_size}KB")
    
    def allocate(self, filename, size):
        if self.algorithm == "First-Fit":
            return self._first_fit_allocate(filename, size)
        elif self.algorithm == "Best-Fit":
            return self._best_fit_allocate(filename, size)
        elif self.algorithm == "Worst-Fit":
            return self._worst_fit_allocate(filename, size)
    
    def _first_fit_allocate(self, filename, size):
        for i, (start, block_size) in enumerate(self.blocks):
            if block_size >= size:
                # Allocate storage from this block
                self.blocks[i] = (start + size, block_size - size)
                if self.blocks[i][1] == 0:
                    self.blocks.pop(i)
                self.allocated[filename] = (start, size)
                logger.info(f"Allocated {size}KB at position {start} for file {filename} using First-Fit")
                return True
        
        logger.warning(f"Failed to allocate {size}KB for file {filename}: No suitable block found")
        return False
    
    def _best_fit_allocate(self, filename, size):
        best_fit_index = -1
        best_fit_size = float('inf')
        
        for i, (start, block_size) in enumerate(self.blocks):
            if size <= block_size < best_fit_size:
                best_fit_index = i
                best_fit_size = block_size
        
        if best_fit_index != -1:
            start, block_size = self.blocks[best_fit_index]
            self.blocks[best_fit_index] = (start + size, block_size - size)
            if self.blocks[best_fit_index][1] == 0:
                self.blocks.pop(best_fit_index)
            self.allocated[filename] = (start, size)
            logger.info(f"Allocated {size}KB at position {start} for file {filename} using Best-Fit")
            return True
        
        logger.warning(f"Failed to allocate {size}KB for file {filename}: No suitable block found")
        return False
    
    def _worst_fit_allocate(self, filename, size):
        worst_fit_index = -1
        worst_fit_size = -1
        
        for i, (start, block_size) in enumerate(self.blocks):
            if size <= block_size and block_size > worst_fit_size:
                worst_fit_index = i
                worst_fit_size = block_size
        
        if worst_fit_index != -1:
            start, block_size = self.blocks[worst_fit_index]
            self.blocks[worst_fit_index] = (start + size, block_size - size)
            if self.blocks[worst_fit_index][1] == 0:
                self.blocks.pop(worst_fit_index)
            self.allocated[filename] = (start, size)
            logger.info(f"Allocated {size}KB at position {start} for file {filename} using Worst-Fit")
            return True
        
        logger.warning(f"Failed to allocate {size}KB for file {filename}: No suitable block found")
        return False
    
    def deallocate(self, filename):
        if filename in self.allocated:
            start, size = self.allocated.pop(filename)
            self.blocks.append((start, size))
            self._merge_adjacent_blocks()
            logger.info(f"Deallocated {size}KB at position {start} for file {filename}")
            return True
        
        logger.warning(f"Failed to deallocate storage for file {filename}: Not allocated")
        return False
    
    def _merge_adjacent_blocks(self):
        self.blocks.sort()  # Sort blocks by start position
        i = 0
        while i < len(self.blocks) - 1:
            current_start, current_size = self.blocks[i]
            next_start, next_size = self.blocks[i + 1]
            
            if current_start + current_size == next_start:
                # Merge adjacent blocks
                self.blocks[i] = (current_start, current_size + next_size)
                self.blocks.pop(i + 1)
            else:
                i += 1
    
    def compact(self):
        if not self.allocated:
            self.blocks = [(0, self.total_size)]
            logger.info("Storage compacted (no allocated blocks)")
            return
        
        # Sort allocated blocks by start position
        allocated_list = sorted(self.allocated.items(), key=lambda x: x[1][0])
        
        # Create new allocation
        new_allocated = {}
        next_position = 0
        
        for filename, (_, size) in allocated_list:
            new_allocated[filename] = (next_position, size)
            next_position += size
        
        # Update allocated blocks
        self.allocated = new_allocated
        
        # Update free blocks
        if next_position < self.total_size:
            self.blocks = [(next_position, self.total_size - next_position)]
        else:
            self.blocks = []
        
        logger.info("Storage compacted")
    
    def fragmentation(self):
        if not self.blocks:
            return 0
        
        total_free = sum(size for _, size in self.blocks)
        largest_free = max(size for _, size in self.blocks) if self.blocks else 0
        
        if total_free == 0:
            return 0
        
        return (total_free - largest_free) / self.total_size
    
    def change_algorithm(self, algorithm):
        self.algorithm = algorithm
        logger.info(f"Changed storage allocation algorithm to {algorithm}")

# File System Class with enhanced features
class FileSystem:
    def __init__(self, max_size=Config.STORAGE_SIZE):
        self.files = {}  # Maps filename to (content, size, path)
        self.directories = {"/": []}  # Maps directory path to list of contents
        self.storage = Storage(max_size)
        self.max_size = max_size
        self.current_dir = "/"
        logger.info(f"File system initialized with max size {max_size}KB")
    
    def read_file(self, filename):
        """Read the contents of a file."""
        full_path = f"{self.current_dir}{filename}"
        if full_path in self.files:
            content, size, _ = self.files[full_path]
            return content
        return None
    
    def write_file(self, filename, content):
        """Write content to a file. Creates the file if it doesn't exist."""
        full_path = f"{self.current_dir}{filename}"
        size = len(content.encode('utf-8')) // 1024 + 1  # Convert to KB
        
        if full_path in self.files:
            # Update existing file
            old_content, old_size, path = self.files[full_path]
            if self.storage.deallocate(full_path) and self.storage.allocate(full_path, size):
                self.files[full_path] = (content, size, path)
                logger.info(f"Updated file {full_path} with size {size}KB")
                return True
        else:
            # Create new file
            if self.storage.allocate(full_path, size):
                self.files[full_path] = (content, size, self.current_dir)
                self.directories[self.current_dir].append((filename, "file"))
                logger.info(f"Created file {full_path} with size {size}KB")
                return True
        
        logger.warning(f"Failed to write to file {full_path}")
        return False

    def append_file(self, filename, content):
        """Append content to an existing file."""
        full_path = f"{self.current_dir}{filename}"
        if full_path in self.files:
            old_content, old_size, path = self.files[full_path]
            new_content = old_content + content
            return self.write_file(filename, new_content)
        return False
    
    def create_directory(self, dirname):
        """Create a new directory in the current directory."""
        if not dirname or "/" in dirname:
            return False
        
        full_path = f"{self.current_dir}{dirname}/"
        if full_path in self.directories:
            return False
        
        self.directories[full_path] = []
        self.directories[self.current_dir].append((dirname, "dir"))
        logger.info(f"Created directory {full_path}")
        return True
    
    def rename_file(self, old_name, new_name):
        """Rename a file in the current directory."""
        old_path = f"{self.current_dir}{old_name}"
        new_path = f"{self.current_dir}{new_name}"
        
        if old_path not in self.files or new_path in self.files:
            return False
        
        # Get the file data
        content, size, path = self.files[old_path]
        
        # Update the file entry
        del self.files[old_path]
        self.files[new_path] = (content, size, path)
        
        # Update the directory listing
        dir_contents = self.directories[self.current_dir]
        for i, (name, type_) in enumerate(dir_contents):
            if name == old_name and type_ == "file":
                dir_contents[i] = (new_name, "file")
                break
        
        logger.info(f"Renamed file {old_path} to {new_path}")
        return True
    
    def change_directory(self, dirname):
        """Change current directory."""
        if dirname == "..":
            # Move up one level
            if self.current_dir != "/":
                self.current_dir = "/".join(self.current_dir.split("/")[:-2]) + "/"
            return True
        elif dirname == "/":
            # Move to root
            self.current_dir = "/"
            return True
        
        full_path = f"{self.current_dir}{dirname}/"
        if full_path in self.directories:
            self.current_dir = full_path
            return True
        return False
    
    def create_file(self, filename, content):
        """Create a new file in the current directory."""
        size = random.randint(Config.MIN_FILE_SIZE, Config.MAX_FILE_SIZE)
        full_path = f"{self.current_dir}{filename}"
        
        if full_path not in self.files and self.storage.allocate(full_path, size):
            self.files[full_path] = (content, size, self.current_dir)
            self.directories[self.current_dir].append((filename, "file"))
            logger.info(f"Created file {full_path} with size {size}KB")
            return True
        return False
    
    def delete_file(self, filename):
        """Delete a file from the current directory."""
        # Handle both full paths and relative paths
        if filename.startswith("/"):
            full_path = filename
        else:
            full_path = f"{self.current_dir}{filename}"
            
        if full_path in self.files:
            _, size, _ = self.files[full_path]
            del self.files[full_path]
            self.storage.deallocate(full_path)
            
            # Update directory listing
            dir_contents = self.directories[self.current_dir]
            for i, (name, type_) in enumerate(dir_contents):
                if name == filename and type_ == "file":
                    dir_contents.pop(i)
                    break
            
            logger.info(f"Deleted file {full_path}")
            return True
        return False
    
    def delete_directory(self, dirname):
        """Delete a directory if it's empty."""
        full_path = f"{self.current_dir}{dirname}/"
        if full_path in self.directories and not self.directories[full_path]:
            del self.directories[full_path]
            self.directories[self.current_dir].remove((dirname, "dir"))
            logger.info(f"Deleted directory {full_path}")
            return True
        return False
    
    def list_contents(self):
        """List contents of current directory."""
        return self.directories.get(self.current_dir, [])

    def search_files(self, query):
        """Search for files by name."""
        return [fname for fname, _ in self.files.items() if query.lower() in fname.lower()]

# UI Components
class Button:
    def __init__(self, x, y, width, height, text, color, action=None, args=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.action = action
        self.args = args if args is not None else []
    
    def draw(self, screen, font, mouse_pos):
        # Check if mouse is over button
        hover = self.rect.collidepoint(mouse_pos)
        color = Config.DARK_GRAY if hover else self.color
        
        # Draw button
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        text_surf = font.render(self.text, True, Config.WHITE)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos) and self.action:
                if self.args:
                    return self.action(*self.args)
                else:
                    return self.action()
        return False

class Tooltip:
    def __init__(self, text, font):
        self.text = text
        self.font = font
        self.active = False
        self.rect = None
        self.surface = None
        self._create_surface()
    
    def _create_surface(self):
        self.surface = self.font.render(self.text, True, Config.BLACK)
        self.rect = self.surface.get_rect()
    
    def show(self, pos):
        self.active = True
        self.rect.topleft = (pos[0], pos[1] - self.rect.height)
    
    def hide(self):
        self.active = False
    
    def draw(self, screen):
        if self.active:
            # Draw background
            bg_rect = self.rect.inflate(10, 5)
            pygame.draw.rect(screen, Config.WHITE, bg_rect)
            pygame.draw.rect(screen, Config.BLACK, bg_rect, 1)
            
            # Draw text
            screen.blit(self.surface, self.rect)

class TextInput:
    def __init__(self, x, y, width, height, font, initial_text=""):
        self.rect = pygame.Rect(x, y, width, height)
        self.font = font
        self.text = initial_text
        self.active = False
        self.cursor_visible = True
        self.cursor_timer = 0
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                return True  # Signal that Enter was pressed
            elif event.key == pygame.K_ESCAPE:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                self.text += event.unicode
        
        return False
    
    def draw(self, screen):
        # Draw text input box
        pygame.draw.rect(screen, Config.WHITE, self.rect)
        pygame.draw.rect(screen, Config.BLACK if self.active else Config.GRAY, self.rect, 2)
        
        # Draw text
        text_surf = self.font.render(self.text, True, Config.BLACK)
        screen.blit(text_surf, (self.rect.x + 5, self.rect.y + 5))
        
        # Draw cursor if active
        if self.active:
            self.cursor_timer += 1
            if self.cursor_timer // 10 % 2 == 0:  # Blink every 10 frames
                text_width = self.font.size(self.text)[0]
                pygame.draw.line(
                    screen, 
                    Config.BLACK,
                    (self.rect.x + 5 + text_width, self.rect.y + 5),
                    (self.rect.x + 5 + text_width, self.rect.y + self.rect.height - 5)
                )

class ResourceManager:
    def __init__(self, resource_names):
        self.resources = {name: None for name in resource_names}
        self.requests = {}
        self.allocated = {}  # Maps process ID to list of resources it holds

    def request(self, pid, resource):
        """Request a resource for a process"""
        if self.resources[resource] is None:
            # Resource is free, allocate it
            self.resources[resource] = pid
            if pid not in self.allocated:
                self.allocated[pid] = []
            self.allocated[pid].append(resource)
            logger.info(f"Process {pid} acquired resource {resource}")
            return True
        else:
            # Resource is in use, add to requests
            if pid not in self.requests:
                self.requests[pid] = []
            if resource not in self.requests[pid]:
                self.requests[pid].append(resource)
            logger.info(f"Process {pid} waiting for resource {resource}")
            return False

    def release(self, pid):
        """Release all resources held by a process"""
        if pid in self.allocated:
            for resource in self.allocated[pid]:
                self.resources[resource] = None
                logger.info(f"Process {pid} released resource {resource}")
            del self.allocated[pid]
        
        # Remove any pending requests
        if pid in self.requests:
            del self.requests[pid]

    def detect_deadlock(self):
        """Detect deadlock using resource allocation graph."""
        # Build resource allocation graph
        graph = {}
        
        # Add edges for resources each process is holding
        for pid, resources in self.allocated.items():
            if pid not in graph:
                graph[pid] = set()
            for resource in resources:
                if resource not in graph:
                    graph[resource] = set()
                graph[pid].add(resource)  # Process -> Resource edge
        
        # Add edges for resources each process is waiting for
        for pid, resources in self.requests.items():
            if pid not in graph:
                graph[pid] = set()
            for resource in resources:
                if resource not in graph:
                    graph[resource] = set()
                graph[resource].add(pid)  # Resource -> Process edge
        
        # Add edges for IO-bound processes
        for pid, resources in self.allocated.items():
            if pid not in graph:
                graph[pid] = set()
            # Add an edge to represent IO wait
            graph[pid].add("IO")  # Process -> IO edge
            if "IO" not in graph:
                graph["IO"] = set()
            graph["IO"].add(pid)  # IO -> Process edge
        
        # Detect cycles using DFS
        visited = set()
        path = []
        cycles = []
        
        def dfs(node):
            if node in path:
                cycle_start = path.index(node)
                cycle = path[cycle_start:]
                if len(cycle) >= 4:  # Ensure it's a valid deadlock cycle (at least 2 processes and 2 resources)
                    cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, set()):
                dfs(neighbor)
            
            path.pop()
        
        # Start DFS from each node
        for node in graph:
            if node not in visited:
                dfs(node)
        
        if not cycles:
            return False, []
        
        # Convert cycles to process-resource pairs
        deadlock_info = []
        for cycle in cycles:
            cycle_info = []
            for i in range(0, len(cycle), 2):
                if i + 1 < len(cycle):
                    cycle_info.append((cycle[i], cycle[i + 1]))
            deadlock_info.append(cycle_info)
        
        return True, deadlock_info

    def get_deadlocked_processes(self):
        """Get list of processes involved in deadlock"""
        if not self.detect_deadlock():
            return []
        
        # Find processes that are both waiting and holding resources
        deadlocked = []
        for pid in self.requests:
            if pid in self.allocated:
                deadlocked.append(pid)
        return deadlocked

class Dropdown:
    def __init__(self, x, y, width, height, options, font, default_value):
        self.rect = pygame.Rect(x, y, width, height)
        self.options = options
        self.selected = default_value
        self.font = font
        self.active = False
        self.dropdown_rect = pygame.Rect(x, y + height, width, height * len(options))
    
    def draw(self, screen, mouse_pos):
        # Draw main button
        color = Config.DARK_GRAY if self.active else Config.GRAY
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        text = self.font.render(f"FPS: {self.selected}", True, Config.WHITE)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)
        
        # Draw dropdown if active
        if self.active:
            pygame.draw.rect(screen, Config.WHITE, self.dropdown_rect)
            pygame.draw.rect(screen, Config.BLACK, self.dropdown_rect, 1)
            
            for i, option in enumerate(self.options):
                option_rect = pygame.Rect(
                    self.rect.x,
                    self.rect.y + self.rect.height * (i + 1),
                    self.rect.width,
                    self.rect.height
                )
                
                # Highlight if mouse is over option
                if option_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(screen, Config.LIGHT_BLUE, option_rect)
                
                text = self.font.render(str(option), True, Config.BLACK)
                text_rect = text.get_rect(center=option_rect.center)
                screen.blit(text, text_rect)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
                return True
            
            if self.active:
                for i, option in enumerate(self.options):
                    option_rect = pygame.Rect(
                        self.rect.x,
                        self.rect.y + self.rect.height * (i + 1),
                        self.rect.width,
                        self.rect.height
                    )
                    if option_rect.collidepoint(event.pos):
                        self.selected = option
                        self.active = False
        
        # Close dropdown if clicking elsewhere
        if event.type == pygame.MOUSEBUTTONDOWN and self.active:
            if not self.rect.collidepoint(event.pos) and not self.dropdown_rect.collidepoint(event.pos):
                self.active = False
        
        return False

class TextEditor:
    def __init__(self, font):
        self.font = font
        self.content = ""
        self.filename = ""
        self.cursor_pos = 0
        self.scroll_pos = 0
        self.running = False
        self.screen = None
        self.clock = None
        self.save_button = None
        self.close_button = None

    def initialize(self):
        """Initialize the editor window and components."""
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Text Editor")
        self.clock = pygame.time.Clock()
        
        # Create buttons with proper actions
        button_font = pygame.font.SysFont(None, 24)
        self.save_button = Button(650, 550, 100, 30, "Save", Config.GREEN, self.save_file)
        self.close_button = Button(550, 550, 100, 30, "Close", Config.RED, self.close)

    def open(self, filename, content):
        """Open a file in the editor."""
        self.filename = filename
        self.content = content
        self.cursor_pos = len(content)
        self.scroll_pos = 0

    def save_file(self):
        """Save the current file content."""
        if hasattr(self, 'filesystem') and self.filename:
            if self.filesystem.write_file(self.filename, self.content):
                self.status_message = f"Saved changes to {self.filename}"
            else:
                self.status_message = f"Failed to save {self.filename}"

    def handle_events(self, filesystem):
        """Handle editor events."""
        self.filesystem = filesystem  # Store filesystem reference for save operation
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if self.save_button.handle_event(event):
                    self.save_file()
                elif self.close_button.handle_event(event):
                    self.close()
                    return
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    self.content = self.content[:self.cursor_pos] + '\n' + self.content[self.cursor_pos:]
                    self.cursor_pos += 1
                elif event.key == pygame.K_BACKSPACE:
                    if self.cursor_pos > 0:
                        self.content = self.content[:self.cursor_pos-1] + self.content[self.cursor_pos:]
                        self.cursor_pos -= 1
                elif event.key == pygame.K_DELETE:
                    if self.cursor_pos < len(self.content):
                        self.content = self.content[:self.cursor_pos] + self.content[self.cursor_pos+1:]
                elif event.key == pygame.K_LEFT:
                    if self.cursor_pos > 0:
                        self.cursor_pos -= 1
                elif event.key == pygame.K_RIGHT:
                    if self.cursor_pos < len(self.content):
                        self.cursor_pos += 1
                elif event.key == pygame.K_UP:
                    lines = self.content[:self.cursor_pos].split('\n')
                    if len(lines) > 1:
                        prev_line = lines[-2]
                        self.cursor_pos -= len(lines[-1]) + 1
                        self.cursor_pos = max(0, self.cursor_pos - len(prev_line))
                elif event.key == pygame.K_DOWN:
                    lines = self.content[self.cursor_pos:].split('\n')
                    if len(lines) > 1:
                        next_line = lines[1]
                        self.cursor_pos += len(lines[0]) + 1
                        self.cursor_pos = min(len(self.content), self.cursor_pos + len(next_line))
                elif event.key == pygame.K_HOME:
                    lines = self.content[:self.cursor_pos].split('\n')
                    self.cursor_pos -= len(lines[-1])
                elif event.key == pygame.K_END:
                    lines = self.content[:self.cursor_pos].split('\n')
                    self.cursor_pos += len(lines[-1])
                elif event.key == pygame.K_PAGEUP:
                    self.scroll_pos = max(0, self.scroll_pos - 10)
                elif event.key == pygame.K_PAGEDOWN:
                    lines = self.content.split('\n')
                    self.scroll_pos = min(len(lines) - 30, self.scroll_pos + 10)
                elif event.unicode.isprintable():
                    self.content = self.content[:self.cursor_pos] + event.unicode + self.content[self.cursor_pos:]
                    self.cursor_pos += 1

    def run(self, filesystem):
        """Run the editor main loop."""
        self.running = True
        while self.running:
            self.handle_events(filesystem)
            self.draw()
            self.clock.tick(60)
        # Don't call pygame.quit() here as it would close the main application

    def close(self):
        """Close the editor window."""
        self.running = False
        # Don't call pygame.display.quit() as it destroys the display surface

    def draw(self):
        """Draw the editor interface."""
        self.screen.fill(Config.WHITE)
        
        # Draw title
        title = self.font.render(f"Editing: {self.filename}", True, Config.BLACK)
        self.screen.blit(title, (20, 20))
        
        # Draw content area
        pygame.draw.rect(self.screen, Config.LIGHT_BLUE, (20, 60, 760, 480), 2)
        
        # Draw text content
        lines = self.content.split('\n')
        visible_lines = 30  # Approximate number of visible lines
        start_line = self.scroll_pos
        end_line = min(start_line + visible_lines, len(lines))
        
        for i, line in enumerate(lines[start_line:end_line]):
            text = self.font.render(line, True, Config.BLACK)
            self.screen.blit(text, (30, 80 + i * 20))
        
        # Draw cursor
        cursor_line = len(self.content[:self.cursor_pos].split('\n')) - 1
        if start_line <= cursor_line < end_line:
            cursor_x = 30 + self.font.size(self.content[:self.cursor_pos].split('\n')[-1])[0]
            cursor_y = 80 + (cursor_line - start_line) * 20
            pygame.draw.line(self.screen, Config.BLACK, (cursor_x, cursor_y), 
                           (cursor_x, cursor_y + 20), 2)
        
        # Draw buttons
        self.save_button.draw(self.screen, self.font, pygame.mouse.get_pos())
        self.close_button.draw(self.screen, self.font, pygame.mouse.get_pos())
        
        pygame.display.flip()

# OSSim main class (includes drawing on the GUI, functions of various buttons, and event handlers)
class OSSim:
    def __init__(self):
        self.screen, self.clock, self.font = initialize_pygame()
        
        # Add FPS dropdown
        self.fps_dropdown = Dropdown(810, 10, 180, 30, Config.FPS_OPTIONS, self.font, Config.FPS)
        
        self.execution_history = []  # Stores tuples: (pid, start_time, end_time)
        self.last_pid = None
        self.last_start_time = 0

        #CPU CHART
        self.utilization_history = []  # Stores recent utilization values
        self.utilization_history_length = 100  # Limit graph width

        #DEADLOCK SIM
        self.deadlock_simulation_enabled = False
        self.resource_manager = ResourceManager(["Printer", "Disk", "Scanner"])
        self.blocked_processes = []

        # Initialize components
        self.scheduler = Scheduler(algorithm="RR", quantum=2)
        self.memory = Memory(total_size=Config.MEMORY_SIZE)
        self.filesystem = FileSystem(max_size=Config.STORAGE_SIZE)
        self.just_terminated = False   # Flag to track if we just terminated a process
        
        # Simulation state
        self.process_count = 0
        self.running = True
        self.paused = False
        self.status_message = "Welcome to OSSim"
        self.selected_file = None
        self.rename_mode = False
        self.file_scroll = 0
        self.search_mode = False
        self.search_results = []
        
        # UI components
        self.buttons = self._create_buttons()
        self.text_input = TextInput(410, 740, 380, 30, self.font)
        self.tooltips = self._create_tooltips()
        self.active_tooltip = None
        
        # Initialize sounds
        self.terminate_sound = load_sound("terminate.wav")
        self.file_create_sound = load_sound("create.wav")
        self.file_delete_sound = load_sound("delete.wav")
        
        # Performance metrics
        self.cpu_utilization = 0
        self.avg_turnaround_time = 0
        self.avg_waiting_time = 0
        
        logger.info("OSSim initialized")
        
        self.directory_scroll = 0
        self.selected_directory_item = None
        
        # Text editor state
        self.editor_active = False
        self.edited_file = None
        self.editor_content = ""
        self.editor_cursor = 0
        self.editor_scroll = 0
        
        # Initialize text editor
        self.text_editor = TextEditor(self.font)
        self.deadlock_visualizer = None
        
    def _create_buttons(self):
        return [
        Button(10, 770, 150, 30, "Toggle Deadlock", Config.ORANGE, self.toggle_deadlock),
        Button(170, 770, 150, 30, "Resolve Deadlock", Config.RED, self.resolve_deadlock),
        Button(10, 650, 150, 30, "Add Process", Config.DARK_BLUE, self.create_process),
        Button(170, 650, 150, 30, "Step", Config.YELLOW, self.step),
        Button(330, 650, 150, 30, "Create File", Config.LIGHT_GREEN, self.create_file),
        Button(490, 650, 150, 30, "Delete File", Config.RED, self.delete_file),
        Button(650, 650, 150, 30, "Toggle Scheduler", Config.ORANGE, self.toggle_scheduler),
        Button(10, 690, 150, 30, "Pause/Resume", Config.GRAY, self.toggle_pause),
        Button(170, 690, 150, 30, "Compact Memory", Config.LIGHT_PURPLE, self.compact_memory),
        Button(330, 690, 150, 30, "Compact Storage", Config.LIGHT_PURPLE, self.compact_storage),
        Button(490, 690, 150, 30, "Search Files", Config.DARK_BLUE, self.toggle_search),
        Button(650, 690, 150, 30, "Toggle Mem Alg", Config.TEAL, self.toggle_memory_algorithm),
        Button(10, 730, 150, 30, "Toggle Stor Alg", Config.TEAL, self.toggle_storage_algorithm),
        Button(170, 730, 150, 30, "Save State", Config.ORANGE, self.save_state),
        Button(330, 730, 150, 30, "Load State", Config.ORANGE, self.load_state),
        Button(490, 730, 150, 30, "Text Editor", Config.BLUE, self.toggle_editor),
        Button(650, 730, 150, 30, "Add IO Process", Config.DARK_BLUE, lambda: self.create_process(True)),
        Button(330, 770, 150, 30, "Reset Simulator", Config.RED, self.reset_simulator),
        Button(650, 770, 150, 30, "Create Directory", Config.LIGHT_GREEN, self.create_directory),
        Button(490, 770, 150, 30, "Rename File", Config.BLUE, self.rename_selected_file),
        Button(820, 770, 150, 30, "Show Deadlock", Config.PURPLE, self.show_deadlock_visualization)
    ]
    
    def _create_tooltips(self):
        return {
            "Add Process": Tooltip("Create a new CPU-bound process", self.font),
            "Step": Tooltip("Advance simulation by one time unit", self.font),
            "Create File": Tooltip("Create a new file in storage", self.font),
            "Delete File": Tooltip("Delete the selected file", self.font),
            "Toggle Scheduler": Tooltip("Switch between scheduling algorithms", self.font),
            "Pause/Resume": Tooltip("Pause or resume simulation", self.font),
            "Compact Mem": Tooltip("Defragment memory space", self.font),
            "Compact Stor": Tooltip("Defragment storage space", self.font),
            "Search Files": Tooltip("Search for files by name", self.font),
            "Toggle Mem Alg": Tooltip("Switch between memory allocation algorithms", self.font),
            "Toggle Stor Alg": Tooltip("Switch between storage allocation algorithms", self.font),
            "Save State": Tooltip("Save current simulation state", self.font),
            "Load State": Tooltip("Load saved simulation state", self.font),
            "Text Editor": Tooltip("Open text editor for .txt files", self.font),
            "Add IO Process": Tooltip("Create a new IO-bound process", self.font),
            "Reset Simulator": Tooltip("Reset the simulator to initial state", self.font),
            "Create Directory": Tooltip("Create a new directory in the current location", self.font),
            "Rename File": Tooltip("Rename the selected file", self.font),
            "Show Deadlock": Tooltip("Show deadlock visualization", self.font)
        }

    def rename_selected_file(self):
        """Rename the currently selected file."""
        if self.selected_file:
            self.rename_mode = True
            self.text_input.active = True
            self.text_input.text = self.selected_file.split("/")[-1]  # Show just the filename
            self.status_message = "Enter new file name"
        else:
            self.status_message = "No file selected!"

    def create_process(self, io_bound=False):
        """Create a new process and add it to the scheduler."""
        self.process_count += 1
        burst_time = random.randint(Config.MIN_BURST_TIME, Config.MAX_BURST_TIME)
        priority = random.randint(Config.MIN_PRIORITY, Config.MAX_PRIORITY)
        proc = Process(self.process_count, self.scheduler.current_time, burst_time, priority, io_bound)
        
        # Try to allocate memory
        memory_size = burst_time * 10
        if self.memory.allocate(proc, memory_size):
            self.scheduler.add_process(proc)
            self.status_message = f"Process {proc.pid} ({priority}) created" + (" (IO-bound)" if io_bound else "")
        else:
            self.status_message = "Not enough memory to create process!"
    
    def step(self):
        """Advance the simulation by one time unit."""
        try:
            previous_terminated_count = len(self.scheduler.terminated)

            # Handle IO processes first
            for i in range(len(self.scheduler.io_queue)-1, -1, -1):
                process = self.scheduler.io_queue[i]
                process.io_wait_time -= 1
                if process.io_wait_time <= 0:
                    process.state = "Ready"
                    self.scheduler.ready_queue.append(process)
                    self.scheduler.io_queue.pop(i)
                    logger.info(f"Process {process.pid} returned from IO")
                    # Update Gantt chart for IO completion
                    if self.last_pid is not None:
                        self.execution_history.append((self.last_pid, self.last_start_time, self.scheduler.current_time))
                    self.last_pid = process.pid
                    self.last_start_time = self.scheduler.current_time

            # Select a process to run if none is running
            if not self.scheduler.current_process:
                # Filter out blocked processes before selecting
                self.scheduler.ready_queue = [p for p in self.scheduler.ready_queue if p.state != "Blocked"]

                if not self.scheduler.ready_queue:
                    return  # No ready process to run

                # Select process based on scheduling algorithm
                if self.scheduler.algorithm == "FCFS":
                    self.scheduler.current_process = self.scheduler.ready_queue.pop(0)
                elif self.scheduler.algorithm == "RR":
                    self.scheduler.current_process = self.scheduler.ready_queue.pop(0)
                    self.scheduler.time_slice = 0
                elif self.scheduler.algorithm == "SJF":
                    self.scheduler.ready_queue.sort(key=lambda p: p.remaining_time)
                    self.scheduler.current_process = self.scheduler.ready_queue.pop(0)
                elif self.scheduler.algorithm == "Priority":
                    self.scheduler.ready_queue.sort(key=lambda p: p.priority, reverse=True)
                    self.scheduler.current_process = self.scheduler.ready_queue.pop(0)

                if self.scheduler.current_process:
                    self.scheduler.current_process.state = "Running"
                    logger.info(f"Selected process {self.scheduler.current_process.pid} to run")
                    # Update Gantt chart for new process
                    if self.last_pid is not None:
                        self.execution_history.append((self.last_pid, self.last_start_time, self.scheduler.current_time))
                    self.last_pid = self.scheduler.current_process.pid
                    self.last_start_time = self.scheduler.current_time

            # Process execution
            if self.scheduler.current_process:
                if self.scheduler.current_process.state == "Blocked":
                    return  # Don't run blocked processes

                # Update process execution time
                self.scheduler.current_process.remaining_time -= 1
                self.scheduler.current_time += 1
                self.scheduler.time_slice += 1
                
                # Handle IO requests
                if (self.scheduler.current_process.io_bound and 
                    random.random() < 0.2 and 
                    self.scheduler.current_process.remaining_time > 0):
                    self.scheduler.current_process.state = "IO"
                    self.scheduler.current_process.io_wait_time = random.randint(1, 5)
                    self.scheduler.io_queue.append(self.scheduler.current_process)
                    logger.info(f"Process {self.scheduler.current_process.pid} requested IO, waiting for {self.scheduler.current_process.io_wait_time}")
                    # Update Gantt chart for IO start
                    if self.last_pid is not None:
                        self.execution_history.append((self.last_pid, self.last_start_time, self.scheduler.current_time))
                    self.last_pid = None
                    self.scheduler.current_process = None
                    return
                
                # Deadlock simulation logic
                if self.deadlock_simulation_enabled:
                    proc = self.scheduler.current_process
                    if not hasattr(proc, "held_resources"):
                        proc.held_resources = []

                    # Only apply deadlock simulation to IO-bound processes
                    if proc.io_bound:
                        # If process is not holding any resources, it must acquire one first
                        if not proc.held_resources:
                            # Try to acquire any available resource
                            available_resources = ["Printer", "Disk", "Scanner"]
                            for resource in available_resources:
                                if self.resource_manager.request(proc.pid, resource):
                                    proc.held_resources.append(resource)
                                    logger.info(f"Process {proc.pid} acquired resource {resource}")
                                    return
                            
                            # If no resources available, process is blocked
                            proc.state = "Blocked"
                            self.scheduler.ready_queue.append(proc)
                            self.scheduler.current_process = None
                            self.blocked_processes.append(proc)
                            logger.info(f"Process {proc.pid} blocked - no resources available")
                            # Update Gantt chart for blocking
                            if self.last_pid is not None:
                                self.execution_history.append((self.last_pid, self.last_start_time, self.scheduler.current_time))
                            self.last_pid = None
                            return
                        
                        # If process is holding a resource, it can request another one
                        elif random.random() < 0.3:  # 30% chance to request another resource
                            available_resources = ["Printer", "Disk", "Scanner"]
                            current_resource = proc.held_resources[0]
                            
                            # Try to request a different resource
                            for resource in available_resources:
                                if resource != current_resource:  # Don't request the same resource
                                    for other_pid, resources in self.resource_manager.allocated.items():
                                        if resource in resources and other_pid != proc.pid:
                                            if self.resource_manager.request(proc.pid, resource):
                                                proc.held_resources.append(resource)
                                                logger.info(f"Process {proc.pid} acquired additional resource {resource}")
                                            else:
                                                proc.state = "Blocked"
                                                self.scheduler.ready_queue.append(proc)
                                                self.scheduler.current_process = None
                                                self.blocked_processes.append(proc)
                                                logger.info(f"Process {proc.pid} blocked requesting {resource} while holding {current_resource}")
                                                # Update Gantt chart for blocking
                                                if self.last_pid is not None:
                                                    self.execution_history.append((self.last_pid, self.last_start_time, self.scheduler.current_time))
                                                self.last_pid = None
                                                return
                        
                        # Process can release its resource
                        elif random.random() < 0.2:  # 20% chance to release the resource
                            resource = proc.held_resources[0]
                            self.resource_manager.release(proc.pid)
                            proc.held_resources.remove(resource)
                            logger.info(f"Process {proc.pid} released resource {resource}")

                # Check if process completed
                if self.scheduler.current_process.remaining_time <= 0:
                    self.scheduler.current_process.state = "Terminated"
                    self.scheduler.current_process.completion_time = self.scheduler.current_time
                    self.scheduler.terminated.append(self.scheduler.current_process)
                    
                    # Clean up memory and resources for completed process
                    if self.scheduler.current_process in self.memory.allocated:
                        start, size = self.memory.allocated[self.scheduler.current_process]
                        self.memory.blocks.append((start, size))
                        del self.memory.allocated[self.scheduler.current_process]
                        self.memory._merge_adjacent_blocks()
                    
                    self.resource_manager.release(self.scheduler.current_process.pid)
                    logger.info(f"Process {self.scheduler.current_process.pid} terminated at time {self.scheduler.current_time}")
                    # Update Gantt chart for termination
                    if self.last_pid is not None:
                        self.execution_history.append((self.last_pid, self.last_start_time, self.scheduler.current_time))
                    self.last_pid = None
                    self.scheduler.current_process = None
                    
                    # Play termination sound and update status
                    if self.terminate_sound:
                        self.terminate_sound.play()
                    self.status_message = f"Process {self.scheduler.terminated[-1].pid} terminated"
                    self.update_performance_metrics()
                # Check if time quantum expired for Round Robin
                elif self.scheduler.algorithm == "RR" and self.scheduler.time_slice >= self.scheduler.quantum:
                    self.scheduler.current_process.state = "Ready"
                    self.scheduler.ready_queue.append(self.scheduler.current_process)
                    logger.info(f"Process {self.scheduler.current_process.pid} time quantum expired, returning to ready queue")
                    # Update Gantt chart for quantum expiration
                    if self.last_pid is not None:
                        self.execution_history.append((self.last_pid, self.last_start_time, self.scheduler.current_time))
                    self.last_pid = None
                    self.scheduler.current_process = None
                    self.scheduler.time_slice = 0

            # Track process execution for Gantt chart
            if self.scheduler.current_process:
                current_pid = self.scheduler.current_process.pid
                if self.last_pid != current_pid:
                    if self.last_pid is not None:
                        self.execution_history.append((self.last_pid, self.last_start_time, self.scheduler.current_time))
                    self.last_pid = current_pid
                    self.last_start_time = self.scheduler.current_time
            elif self.last_pid is not None:
                self.execution_history.append((self.last_pid, self.last_start_time, self.scheduler.current_time))
                self.last_pid = None

            # Handle termination
            if len(self.scheduler.terminated) > previous_terminated_count:
                terminated_proc = self.scheduler.terminated[-1]
                self.memory.deallocate(terminated_proc)
                self.resource_manager.release(terminated_proc.pid)
                self.status_message = f"Process {terminated_proc.pid} terminated"

                if self.terminate_sound:
                    self.terminate_sound.play()

                self.update_performance_metrics()

            # Check for deadlock
            if self.deadlock_simulation_enabled:
                deadlock_detected, deadlock_info = self.resource_manager.detect_deadlock()
                if deadlock_detected:
                    status_parts = ["⚠️ Deadlock detected!"]
                    for cycle in deadlock_info:
                        for pid, resource in cycle:
                            # Get what the process is holding
                            holding = []
                            for held_pid, resources in self.resource_manager.allocated.items():
                                if held_pid == pid:
                                    holding.extend(resources)
                            
                            # Get what the process is waiting for
                            waiting_for = []
                            for req_pid, resources in self.resource_manager.requests.items():
                                if req_pid == pid:
                                    waiting_for.extend(resources)
                            
                            status_parts.append(f"PID {pid}: Holding {holding if holding else 'nothing'}, Waiting for {waiting_for if waiting_for else 'nothing'}")
                    self.status_message = "\n".join(status_parts)

        except Exception as e:
            logger.error(f"Error in step: {e}")
            self.status_message = f"Error: {str(e)}"

    def create_file(self):
        """Create a new file in the file system with enhanced features."""
        # Generate random file type
        file_types = {
            'txt': 'Sample text content',
            'log': 'Log entry ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'jpg': 'Image data',
            'png': 'Image data',
            'mp3': 'Audio data'
        }
        
        # Randomly select file type
        file_ext = random.choice(list(file_types.keys()))
        fname = f"file{len(self.filesystem.files)+1}.{file_ext}"
        
        if self.filesystem.create_file(fname, file_types[file_ext]):
            self.status_message = f"File {fname} created ({file_ext.upper()} - {file_types[file_ext]})"
            if self.file_create_sound:
                self.file_create_sound.play()
        else:
            self.status_message = "Storage full or file already exists!"
    
    def delete_file(self):
        """Delete the selected file from the file system."""
        if self.selected_file:
            # Extract just the filename from the full path
            filename = self.selected_file.split("/")[-1]
            
            if self.filesystem.delete_file(filename):
                self.status_message = f"File {self.selected_file} deleted"
                if self.file_delete_sound:
                    self.file_delete_sound.play()
                self.selected_file = None
            else:
                self.status_message = f"Failed to delete {self.selected_file}"
        else:
            self.status_message = "No file selected!"
    
    def toggle_scheduler(self):
        """Cycle through available scheduling algorithms."""
        scheduler_options = ["RR", "FCFS", "SJF", "Priority"]
        current_idx = scheduler_options.index(self.scheduler.algorithm)
        next_idx = (current_idx + 1) % len(scheduler_options)
        new_algorithm = scheduler_options[next_idx]
        self.scheduler.change_algorithm(new_algorithm)
        self.status_message = f"Scheduler set to {new_algorithm}"
    
    def toggle_pause(self):
        """Toggle the simulation pause state."""
        self.paused = not self.paused
        self.status_message = "Paused" if self.paused else "Resumed"
    
    def compact_memory(self):
        """Defragment the memory space."""
        self.memory.compact()
        self.status_message = "Memory compacted"
    
    def compact_storage(self):
        """Defragment the storage space."""
        self.filesystem.storage.compact()
        self.status_message = "Storage compacted"
    
    def toggle_search(self):
        """Toggle search mode for files."""
        self.search_mode = not self.search_mode
        if self.search_mode:
            self.status_message = "Enter search term"
            self.text_input.active = True
            self.text_input.text = ""
        else:
            self.search_results = []
            self.status_message = "Search mode disabled"
    
    def toggle_memory_algorithm(self):
        """Cycle through memory allocation algorithms."""
        memory_algorithms = ["First-Fit", "Best-Fit", "Worst-Fit"]
        current_idx = memory_algorithms.index(self.memory.algorithm)
        next_idx = (current_idx + 1) % len(memory_algorithms)
        new_algorithm = memory_algorithms[next_idx]
        self.memory.change_algorithm(new_algorithm)
        self.status_message = f"Memory algorithm set to {new_algorithm}"
    
    def toggle_storage_algorithm(self):
        """Cycle through storage allocation algorithms."""
        storage_algorithms = ["First-Fit", "Best-Fit", "Worst-Fit"]
        current_idx = storage_algorithms.index(self.filesystem.storage.algorithm)
        next_idx = (current_idx + 1) % len(storage_algorithms)
        new_algorithm = storage_algorithms[next_idx]
        self.filesystem.storage.change_algorithm(new_algorithm)
        self.status_message = f"Storage algorithm set to {new_algorithm}"
    
    def save_state(self):

        try:
            state = {
            'scheduler': self.scheduler,
            'memory': self.memory,
            'filesystem': self.filesystem,
            'process_count': self.process_count,
            'current_time': self.scheduler.current_time
            }
            with open("ossim_state.pkl", "wb") as f:
                pickle.dump(state, f)
            self.status_message = "State saved successfully"
            logger.info("State saved to ossim_state.pkl")
        except Exception as e:
            self.status_message = "Failed to save state"
            logger.error(f"Save state error: {e}")

    
    def load_state(self):
        try:
            with open("ossim_state.pkl", "rb") as f:
                state = pickle.load(f)
            self.scheduler = state['scheduler']
            self.memory = state['memory']
            self.filesystem = state['filesystem']
            self.process_count = state['process_count']
            self.scheduler.current_time = state['current_time']
            self.status_message = "State loaded successfully"
            logger.info("State loaded from ossim_state.pkl")
        except Exception as e:
            self.status_message = "Failed to load state"
            logger.error(f"Load state error: {e}")

    
    def export_log(self):
        """Export the simulation log to a file."""
        try:
            # This is a placeholder - in a full implementation,
            # we would export the log to a user-selected location
            self.status_message = "Log export not implemented yet"
            logger.info("Log export requested (not implemented)")
        except Exception as e:
            self.status_message = "Failed to export log"
            logger.error(f"Failed to export log: {e}")
    
    def update_performance_metrics(self):
        """Update performance metrics based on terminated processes."""
        if not self.scheduler.terminated:
            return
        
        total_processes = len(self.scheduler.terminated)
        total_turnaround = sum(p.completion_time - p.arrival_time for p in self.scheduler.terminated)
        total_wait = sum((p.completion_time - p.arrival_time) - p.burst_time for p in self.scheduler.terminated)
        
        self.avg_turnaround_time = total_turnaround / total_processes if total_processes > 0 else 0
        self.avg_waiting_time = total_wait / total_processes if total_processes > 0 else 0
        
        # Calculate CPU utilization
        if self.scheduler.current_time > 0:
            busy_time = sum(p.burst_time for p in self.scheduler.terminated)
            self.cpu_utilization = busy_time / self.scheduler.current_time
            self.utilization_history.append(self.cpu_utilization)
            if len(self.utilization_history) > self.utilization_history_length:
                self.utilization_history.pop(0)

    def toggle_deadlock(self):
        self.deadlock_simulation_enabled = not self.deadlock_simulation_enabled
        state = "enabled" if self.deadlock_simulation_enabled else "disabled"
        self.status_message = f"Deadlock simulation {state}"

    def resolve_deadlock(self):
        """Resolve deadlock by terminating an IO-bound process involved in the deadlock."""
        if not self.deadlock_simulation_enabled:
            return

        # Get deadlock information
        deadlock_detected, deadlock_info = self.resource_manager.detect_deadlock()
        
        if not deadlock_detected:
            self.status_message = "No deadlock detected"
            return

        # Get all IO-bound processes involved in the deadlock
        io_bound_deadlocked = []
        for cycle in deadlock_info:
            for pid, _ in cycle:
                # Find the process in all queues
                process = None
                for p in (self.scheduler.ready_queue + 
                         self.scheduler.io_queue + 
                         self.blocked_processes + 
                         [self.scheduler.current_process]):
                    if p and p.pid == pid and p.io_bound:
                        process = p
                        break
                
                if process and process not in io_bound_deadlocked:
                    io_bound_deadlocked.append(process)
        
        if not io_bound_deadlocked:
            self.status_message = "No IO-bound processes found in deadlock"
            return

        # Select the process with the lowest priority to terminate
        process_to_terminate = min(io_bound_deadlocked, key=lambda p: p.priority)
        
        # Log the termination
        logger.info(f"Resolving deadlock by terminating IO-bound process {process_to_terminate.pid}")
        
        # Store the resources that will be freed
        freed_resources = []
        if process_to_terminate.pid in self.resource_manager.allocated:
            freed_resources = self.resource_manager.allocated[process_to_terminate.pid].copy()
        
        # Terminate the process and set completion time
        process_to_terminate.state = "Terminated"
        process_to_terminate.completion_time = self.scheduler.current_time
        process_to_terminate.remaining_time = 0  # Ensure process is marked as completed
        
        # Remove from all queues
        for queue in [self.scheduler.ready_queue, 
                     self.scheduler.io_queue, 
                     self.blocked_processes]:
            if process_to_terminate in queue:
                queue.remove(process_to_terminate)
        
        if self.scheduler.current_process == process_to_terminate:
            self.scheduler.current_process = None
        
        # Add to terminated queue
        self.scheduler.terminated.append(process_to_terminate)
        
        # Clean up memory
        if process_to_terminate in self.memory.allocated:
            start, size = self.memory.allocated[process_to_terminate]
            self.memory.blocks.append((start, size))
            del self.memory.allocated[process_to_terminate]
            self.memory._merge_adjacent_blocks()
        
        # Release all resources
        self.resource_manager.release(process_to_terminate.pid)
        
        # Remove from resource manager maps
        if process_to_terminate.pid in self.resource_manager.requests:
            del self.resource_manager.requests[process_to_terminate.pid]
        if process_to_terminate.pid in self.resource_manager.allocated:
            del self.resource_manager.allocated[process_to_terminate.pid]
        
        # Check for processes that can be unblocked
        unblocked_processes = []
        for process in self.blocked_processes[:]:  # Create a copy to safely modify
            if not hasattr(process, "held_resources"):
                continue
                
            # Check if the process can get all its requested resources
            can_unblock = True
            for resource in process.held_resources:
                if self.resource_manager.resources[resource] is not None:
                    can_unblock = False
                    break
            
            if can_unblock:
                # Try to acquire the resources
                success = True
                for resource in process.held_resources:
                    if not self.resource_manager.request(process.pid, resource):
                        success = False
                        break
                
                if success:
                    process.state = "Ready"
                    self.scheduler.ready_queue.append(process)
                    self.blocked_processes.remove(process)
                    unblocked_processes.append(process)
                    logger.info(f"Process {process.pid} unblocked after resource release")
                    
                    # If no process is currently running, schedule this one
                    if not self.scheduler.current_process:
                        self.scheduler.current_process = process
                        self.scheduler.ready_queue.remove(process)
                        process.state = "Running"
                        logger.info(f"Process {process.pid} scheduled to run after unblocking")
                        # Update Gantt chart for newly scheduled process
                        if self.last_pid is not None:
                            self.execution_history.append((self.last_pid, self.last_start_time, self.scheduler.current_time))
                        self.last_pid = process.pid
                        self.last_start_time = self.scheduler.current_time
        
        # Check if there's only one process remaining in memory
        remaining_processes = [p for p in (self.scheduler.ready_queue + 
                                         self.scheduler.io_queue + 
                                         self.blocked_processes + 
                                         [self.scheduler.current_process])
                            if p and p != process_to_terminate]
        
        if len(remaining_processes) == 1:
            # Complete the remaining process
            remaining_process = remaining_processes[0]
            logger.info(f"Completing remaining process {remaining_process.pid}")
            
            # Set remaining time to 0 and completion time
            remaining_process.remaining_time = 0
            remaining_process.completion_time = self.scheduler.current_time
            
            # Update Gantt chart
            if self.last_pid is not None:
                self.execution_history.append((self.last_pid, self.last_start_time, self.scheduler.current_time))
            self.last_pid = remaining_process.pid
            self.last_start_time = self.scheduler.current_time
            
            # Clean up resources and memory
            self.resource_manager.release(remaining_process.pid)
            if remaining_process in self.memory.allocated:
                self.memory.deallocate(remaining_process)
            
            # Add to terminated list
            self.scheduler.terminated.append(remaining_process)
            
            # Remove from any queues
            if remaining_process in self.scheduler.ready_queue:
                self.scheduler.ready_queue.remove(remaining_process)
            if remaining_process in self.scheduler.io_queue:
                self.scheduler.io_queue.remove(remaining_process)
            if remaining_process in self.blocked_processes:
                self.blocked_processes.remove(remaining_process)
            
            # Update status message to include completion of remaining process
            status_parts = [
                f"Deadlock resolved by terminating process {process_to_terminate.pid}",
                f"Process {remaining_process.pid} completed and terminated",
                f"All processes have been completed"
            ]
            self.status_message = "\n".join(status_parts)
        else:
            # Update status message with detailed information
            status_parts = [f"Deadlock resolved by terminating IO-bound process {process_to_terminate.pid}"]
            
            # Add information about what resources were held and requested
            holding = []
            for held_pid, resources in self.resource_manager.allocated.items():
                if held_pid == process_to_terminate.pid:
                    holding.extend(resources)
            
            waiting_for = []
            for req_pid, resources in self.resource_manager.requests.items():
                if req_pid == process_to_terminate.pid:
                    waiting_for.extend(resources)
            
            status_parts.append(f"Holding: {holding if holding else 'none'}")
            status_parts.append(f"Waiting for: {waiting_for if waiting_for else 'none'}")
            
            # Add information about unblocked processes
            if unblocked_processes:
                status_parts.append("\nUnblocked processes:")
                for process in unblocked_processes:
                    status_parts.append(f"Process {process.pid} is now ready")
                    if process == self.scheduler.current_process:
                        status_parts.append(f"Process {process.pid} is now running")
            
            self.status_message = "\n".join(status_parts)
        
        # Play termination sound
        if self.terminate_sound:
            self.terminate_sound.play()
        
        # Update performance metrics
        self.update_performance_metrics()
        
        # Unpause the simulation
        self.paused = False

    def reset_simulator(self):
        """Reset the simulator to its initial state."""
        # Reset scheduler
        self.scheduler = Scheduler(algorithm="RR", quantum=2)
        
        # Reset memory
        self.memory = Memory(total_size=Config.MEMORY_SIZE)
        
        # Reset file system
        self.filesystem = FileSystem(max_size=Config.STORAGE_SIZE)
        
        # Reset process count
        self.process_count = 0
        
        # Reset simulation state
        self.running = True
        self.paused = False
        self.status_message = "Simulator reset to initial state"
        self.selected_file = None
        self.rename_mode = False
        self.file_scroll = 0
        self.search_mode = False
        self.search_results = []
        
        # Reset performance metrics
        self.cpu_utilization = 0
        self.avg_turnaround_time = 0
        self.avg_waiting_time = 0
        self.utilization_history = []
        
        # Reset execution history
        self.execution_history = []
        self.last_pid = None
        self.last_start_time = 0
        
        # Reset deadlock simulation
        self.deadlock_simulation_enabled = False
        self.resource_manager = ResourceManager(["Printer", "Disk", "Scanner"])
        self.blocked_processes = []
        
        logger.info("Simulator reset to initial state")

    def draw_performance_metrics(self, screen):
        """Draw performance metrics in the dashboard."""
        metrics = [
            f"CPU Util: {self.cpu_utilization:.2%}",
            f"Avg Turnaround: {self.avg_turnaround_time:.2f}",
            f"Avg Wait: {self.avg_waiting_time:.2f}"
        ]
        
        for i, metric in enumerate(metrics):
            text = self.font.render(metric, True, Config.BLACK)
            screen.blit(text, (810, 230 + i * 30))
    
    def draw_process(self, screen, process, y, width=120, height=30):
        """Draw a process rectangle with its details."""
        # Choose color based on process state
        state_colors = {
            "Ready": Config.BLUE,
            "Running": Config.GREEN,
            "Terminated": Config.GRAY,
            "IO": Config.ORANGE
        }
        color = state_colors.get(process.state, process.color)
        
        # Transition animation for moving process
        if process.state == "Running" and process.x < 160:
            process.x += 5
        elif process.state != "Running" and process.x > 20:
            process.x -= 5
        
        # Draw process rectangle
        pygame.draw.rect(screen, color, (process.x, y, width, height))
        
        # Draw process details
        text = f"PID: {process.pid} ({process.remaining_time})"
        if process.io_bound:
            text += " IO"
        
        # Add priority indicator
        priority_indicator = "!" * process.priority
        
        text_surf = self.font.render(text, True, Config.BLACK)
        screen.blit(text_surf, (process.x + 5, y + 5))
        
        # Draw priority indicator
        priority_surf = self.font.render(priority_indicator, True, Config.RED)
        screen.blit(priority_surf, (process.x + width - 30, y + 5))
    
    def draw_memory(self, screen, x, y, width, height):
        """Draw memory visualization with detailed information."""
        # Draw main memory container
        pygame.draw.rect(screen, Config.GRAY, (x, y, width, height), 2)
        
        # Draw title and algorithm info
        title_text = self.font.render(f"Memory ({self.memory.algorithm})", True, Config.BLACK)
        screen.blit(title_text, (x + 10, y + 10))
        
        # Add algorithm explanation
        algo_explanation = {
            "First-Fit": "Finds first block that fits",
            "Best-Fit": "Finds smallest block that fits",
            "Worst-Fit": "Finds largest block available"
        }[self.memory.algorithm]
        
        explanation_text = self.font.render(algo_explanation, True, Config.BLUE)
        screen.blit(explanation_text, (x + width - 200, y + 10))
        
        # Draw memory blocks
        memory_height = height - 120  # Increased space for information
        scale_factor = memory_height / self.memory.total_size
        
        # Draw memory scale
        scale_y = y + 40
        pygame.draw.line(screen, Config.BLACK, (x + 10, scale_y), (x + 110, scale_y))
        for i in range(0, self.memory.total_size + 1, 256):  # Show scale every 256KB
            scale_pos = scale_y + int(i * scale_factor)
            pygame.draw.line(screen, Config.BLACK, (x + 8, scale_pos), (x + 12, scale_pos))
            if i % 512 == 0:  # Show labels every 512KB
                scale_text = self.font.render(f"{i}KB", True, Config.BLACK)
                screen.blit(scale_text, (x + 115, scale_pos - 10))
        
        # Draw free blocks with different patterns based on algorithm
        for start, size in self.memory.blocks:
            block_y = scale_y + int(start * scale_factor)
            block_height = max(1, int(size * scale_factor))
            
            # Different patterns for different algorithms
            if self.memory.algorithm == "First-Fit":
                # First Fit: Solid color with diagonal lines
                pygame.draw.rect(screen, Config.RED, (x + 10, block_y, 100, block_height))
                for i in range(0, block_height, 5):
                    pygame.draw.line(screen, (200, 0, 0), 
                                   (x + 10, block_y + i), 
                                   (x + 10 + min(100, i), block_y + i))
            elif self.memory.algorithm == "Best-Fit":
                # Best Fit: Checkered pattern
                pygame.draw.rect(screen, Config.RED, (x + 10, block_y, 100, block_height))
                for i in range(0, block_height, 10):
                    for j in range(0, 100, 10):
                        if (i + j) % 20 == 0:
                            pygame.draw.rect(screen, (200, 0, 0), 
                                          (x + 10 + j, block_y + i, 10, 10))
            else:  # Worst-Fit
                # Worst Fit: Gradient pattern
                pygame.draw.rect(screen, Config.RED, (x + 10, block_y, 100, block_height))
                for i in range(0, block_height, 2):
                    color = (200 - int(i * 100 / block_height), 0, 0)
                    pygame.draw.line(screen, color, 
                                   (x + 10, block_y + i), 
                                   (x + 110, block_y + i))
        
        # Draw allocated blocks (green with process info)
        for proc, (start, size) in self.memory.allocated.items():
            block_y = scale_y + int(start * scale_factor)
            block_height = max(1, int(size * scale_factor))
            
            # Determine block color based on process state
            if proc.state == "Running":
                color = Config.GREEN
            elif proc.state == "Ready":
                color = (0, 200, 0)  # Darker green
            elif proc.state == "Blocked":
                color = (128, 0, 128)  # Violet
            elif proc.state == "IO":
                color = (0, 100, 0)  # Darkest green
            else:
                color = Config.GREEN
            
            # Draw block with border
            pygame.draw.rect(screen, color, (x + 10, block_y, 100, block_height))
            pygame.draw.rect(screen, Config.BLACK, (x + 10, block_y, 100, block_height), 1)
            
            # Draw process information if block is large enough
            if block_height >= 25:  # Increased minimum height for text
                # PID and size
                pid_text = self.font.render(f"PID: {proc.pid}", True, Config.BLACK)
                size_text = self.font.render(f"{size}KB", True, Config.BLACK)
                
                # Calculate text positions to avoid overlap
                pid_y = block_y + 2
                size_y = block_y + block_height - 15
                
                # Only draw if there's enough space between texts
                if size_y - pid_y >= 20:
                    screen.blit(pid_text, (x + 15, pid_y))
                    screen.blit(size_text, (x + 15, size_y))
                else:
                    # If not enough space, just show PID
                    screen.blit(pid_text, (x + 15, block_y + (block_height - pid_text.get_height()) // 2))
                
                # State indicator
                state_color = {
                    "Running": Config.YELLOW,
                    "Ready": Config.BLUE,
                    "Blocked": (128, 0, 128),  # Violet
                    "IO": Config.ORANGE
                }.get(proc.state, Config.WHITE)
                
                pygame.draw.circle(screen, state_color, (x + 100, block_y + 10), 5)
        
        # Draw statistics section
        stats_y = y + height - 80
        pygame.draw.line(screen, Config.GRAY, (x + 10, stats_y), (x + 110, stats_y))  # Reduced line width to match memory blocks
        
        # Draw fragmentation info
        frag = self.memory.fragmentation()
        frag_text = self.font.render(f"Fragmentation: {frag:.2%}", True, Config.BLACK)
        screen.blit(frag_text, (x + 10, stats_y + 5))
        
        # Draw memory usage
        used = sum(size for _, size in self.memory.allocated.values())
        usage_text = self.font.render(f"Used: {used}/{self.memory.total_size}KB", True, Config.BLACK)
        screen.blit(usage_text, (x + 10, stats_y + 25))
        
        # Draw process count
        proc_count = len(self.memory.allocated)
        count_text = self.font.render(f"Active Processes: {proc_count}", True, Config.BLACK)
        screen.blit(count_text, (x + 10, stats_y + 45))
        
        # Create a smaller font for legends
        small_font = pygame.font.SysFont(None, 16)
        
        # Draw legends on the right side
        legend_x = x + width - 120
        legend_y = y + 40
        
        # Draw algorithm patterns legend
        algo_legend = [
            ("First-Fit", "Diagonal lines"),
            ("Best-Fit", "Checkered"),
            ("Worst-Fit", "Gradient")
        ]
        
        # Draw process state colors legend
        process_legend = [
            ("Running", Config.GREEN),
            ("Ready", (0, 200, 0)),
            ("Blocked", (128, 0, 128)),
            ("IO", (0, 100, 0))
        ]
        
        # Draw vertical separator line
        pygame.draw.line(screen, Config.GRAY, (legend_x - 10, legend_y), 
                        (legend_x - 10, legend_y + (len(algo_legend) + len(process_legend)) * 20))
        
        # Draw algorithm patterns
        for i, (algo, pattern) in enumerate(algo_legend):
            item_y = legend_y + i * 20
            # Draw pattern example
            if algo == "First-Fit":
                pygame.draw.rect(screen, Config.RED, (legend_x, item_y, 15, 15))
                pygame.draw.line(screen, (200, 0, 0), (legend_x, item_y), 
                               (legend_x + 15, item_y + 15))
            elif algo == "Best-Fit":
                pygame.draw.rect(screen, Config.RED, (legend_x, item_y, 15, 15))
                pygame.draw.rect(screen, (200, 0, 0), (legend_x, item_y, 7, 7))
                pygame.draw.rect(screen, (200, 0, 0), (legend_x + 8, item_y + 8, 7, 7))
            else:  # Worst-Fit
                pygame.draw.rect(screen, Config.RED, (legend_x, item_y, 15, 15))
                for j in range(0, 15, 2):
                    color = (200 - int(j * 100 / 15), 0, 0)
                    pygame.draw.line(screen, color, 
                                   (legend_x, item_y + j), 
                                   (legend_x + 15, item_y + j))
            
            # Draw text with smaller font
            text = f"{algo}: {pattern}"
            text_surf = small_font.render(text, True, Config.BLACK)
            screen.blit(text_surf, (legend_x + 20, item_y))
        
        # Draw process state colors
        for i, (state, color) in enumerate(process_legend):
            item_y = legend_y + (len(algo_legend) + i) * 20
            # Draw color box
            pygame.draw.rect(screen, color, (legend_x, item_y, 15, 15))
            pygame.draw.rect(screen, Config.BLACK, (legend_x, item_y, 15, 15), 1)
            
            # Draw text with smaller font
            text_surf = small_font.render(state, True, Config.BLACK)
            screen.blit(text_surf, (legend_x + 20, item_y))

    def draw_storage(self, screen, x, y, width, height):
        """Draw the storage visualization with enhanced information."""
        # Draw main storage container
        pygame.draw.rect(screen, Config.GRAY, (x, y, width, height), 2)
        
        # Draw title and algorithm info with explanation
        title_text = self.font.render(f"Storage ({self.filesystem.storage.algorithm})", True, Config.BLACK)
        screen.blit(title_text, (x + 10, y + 10))
        
        # Add algorithm explanation
        algo_explanation = {
            "First-Fit": "Finds first block that fits",
            "Best-Fit": "Finds smallest block that fits",
            "Worst-Fit": "Finds largest block available"
        }[self.filesystem.storage.algorithm]
        
        explanation_text = self.font.render(algo_explanation, True, Config.BLUE)
        screen.blit(explanation_text, (x + width - 200, y + 10))
        
        # Draw storage blocks
        storage_height = height - 120
        scale_factor = storage_height / self.filesystem.storage.total_size
        
        # Draw storage scale
        scale_y = y + 40
        pygame.draw.line(screen, Config.BLACK, (x + 10, scale_y), (x + 110, scale_y))
        for i in range(0, self.filesystem.storage.total_size + 1, 256):
            scale_pos = scale_y + int(i * scale_factor)
            pygame.draw.line(screen, Config.BLACK, (x + 8, scale_pos), (x + 12, scale_pos))
            if i % 512 == 0:
                scale_text = self.font.render(f"{i}KB", True, Config.BLACK)
                screen.blit(scale_text, (x + 120, scale_pos - 10))
        
        # Draw free blocks with different patterns based on algorithm
        for start, size in self.filesystem.storage.blocks:
            block_y = scale_y + int(start * scale_factor)
            block_height = max(1, int(size * scale_factor))
            
            # Different patterns for different algorithms
            if self.filesystem.storage.algorithm == "First-Fit":
                # First Fit: Solid color with diagonal lines
                pygame.draw.rect(screen, Config.RED, (x + 10, block_y, 100, block_height))
                for i in range(0, block_height, 5):
                    pygame.draw.line(screen, (200, 0, 0), 
                                   (x + 10, block_y + i), 
                                   (x + 10 + min(100, i), block_y + i))
            elif self.filesystem.storage.algorithm == "Best-Fit":
                # Best Fit: Checkered pattern
                pygame.draw.rect(screen, Config.RED, (x + 10, block_y, 100, block_height))
                for i in range(0, block_height, 10):
                    for j in range(0, 100, 10):
                        if (i + j) % 20 == 0:
                            pygame.draw.rect(screen, (200, 0, 0), 
                                          (x + 10 + j, block_y + i, 10, 10))
            else:  # Worst-Fit
                # Worst Fit: Gradient pattern
                pygame.draw.rect(screen, Config.RED, (x + 10, block_y, 100, block_height))
                for i in range(0, block_height, 2):
                    color = (200 - int(i * 100 / block_height), 0, 0)
                    pygame.draw.line(screen, color, 
                                   (x + 10, block_y + i), 
                                   (x + 110, block_y + i))
        
        # Draw allocated blocks (with file info)
        for fname, (start, size) in self.filesystem.storage.allocated.items():
            block_y = scale_y + int(start * scale_factor)
            block_height = max(1, int(size * scale_factor))
            
            # Determine block color based on file type
            file_ext = fname.split('.')[-1].lower()
            if file_ext in ['txt', 'log']:
                color = (0, 200, 0)  # Green for text files
            elif file_ext in ['jpg', 'png', 'gif']:
                color = (0, 150, 255)  # Blue for images
            elif file_ext in ['mp3', 'wav']:
                color = (255, 150, 0)  # Orange for audio
            else:
                color = Config.GREEN  # Default color
            
            # Draw block with border
            pygame.draw.rect(screen, color, (x + 10, block_y, 100, block_height))
            pygame.draw.rect(screen, Config.BLACK, (x + 10, block_y, 100, block_height), 1)
            
            # Draw file information if block is large enough
            if block_height >= 25:
                # File name and size
                display_name = fname[:8] + ".." if len(fname) > 10 else fname
                name_text = self.font.render(display_name, True, Config.BLACK)
                size_text = self.font.render(f"{size}KB", True, Config.BLACK)
                
                # Calculate text positions to avoid overlap
                name_y = block_y + 2
                size_y = block_y + block_height - 15
                
                # Only draw if there's enough space between texts
                if size_y - name_y >= 20:
                    screen.blit(name_text, (x + 15, name_y))
                    screen.blit(size_text, (x + 15, size_y))
                else:
                    # If not enough space, just show name
                    screen.blit(name_text, (x + 15, block_y + (block_height - name_text.get_height()) // 2))
        
        # Draw statistics section
        stats_y = y + height - 85  
        pygame.draw.line(screen, Config.GRAY, (x + 10, stats_y), (x + 110, stats_y))
        
        # Calculate storage statistics
        total_used = sum(size for _, size in self.filesystem.storage.allocated.values())
        total_free = sum(size for _, size in self.filesystem.storage.blocks)
        frag = self.filesystem.storage.fragmentation()
        
        # Draw storage statistics
        stats = [
            f"Used: {total_used}/{self.filesystem.storage.total_size}KB",
            f"Free: {total_free}KB",
            f"Fragmentation: {frag:.2%}",
            f"Files: {len(self.filesystem.files)}"
        ]
        
        for i, stat in enumerate(stats):
            stat_text = self.font.render(stat, True, Config.BLACK)
            screen.blit(stat_text, (x + 10, stats_y + 5 + i * 20))
        
        # Create a smaller font for legends
        small_font = pygame.font.SysFont(None, 16)
        
        # Draw legends on the right side
        legend_x = x + width - 120
        legend_y = y + 40
        
        # Draw algorithm patterns legend
        algo_legend = [
            ("First-Fit", "Diagonal lines"),
            ("Best-Fit", "Checkered"),
            ("Worst-Fit", "Gradient")
        ]
        
        # Draw file type colors legend
        file_legend = [
            ("Text files", (0, 200, 0)),
            ("Images", (0, 150, 255)),
            ("Audio", (255, 150, 0))
        ]
        
        # Draw vertical separator line
        pygame.draw.line(screen, Config.GRAY, (legend_x - 10, legend_y), 
                        (legend_x - 10, legend_y + (len(algo_legend) + len(file_legend)) * 20))
        
        # Draw algorithm patterns
        for i, (algo, pattern) in enumerate(algo_legend):
            item_y = legend_y + i * 20
            # Draw pattern example
            if algo == "First-Fit":
                pygame.draw.rect(screen, Config.RED, (legend_x, item_y, 15, 15))
                pygame.draw.line(screen, (200, 0, 0), (legend_x, item_y), 
                               (legend_x + 15, item_y + 15))
            elif algo == "Best-Fit":
                pygame.draw.rect(screen, Config.RED, (legend_x, item_y, 15, 15))
                pygame.draw.rect(screen, (200, 0, 0), (legend_x, item_y, 7, 7))
                pygame.draw.rect(screen, (200, 0, 0), (legend_x + 8, item_y + 8, 7, 7))
            else:  # Worst-Fit
                pygame.draw.rect(screen, Config.RED, (legend_x, item_y, 15, 15))
                for j in range(0, 15, 2):
                    color = (200 - int(j * 100 / 15), 0, 0)
                    pygame.draw.line(screen, color, 
                                   (legend_x, item_y + j), 
                                   (legend_x + 15, item_y + j))
            
            # Draw text with smaller font
            text = f"{algo}: {pattern}"
            text_surf = small_font.render(text, True, Config.BLACK)
            screen.blit(text_surf, (legend_x + 20, item_y))
        
        # Draw file type colors
        for i, (file_type, color) in enumerate(file_legend):
            item_y = legend_y + (len(algo_legend) + i) * 20
            # Draw color box
            pygame.draw.rect(screen, color, (legend_x, item_y, 15, 15))
            pygame.draw.rect(screen, Config.BLACK, (legend_x, item_y, 15, 15), 1)
            
            # Draw text with smaller font
            text_surf = small_font.render(file_type, True, Config.BLACK)
            screen.blit(text_surf, (legend_x + 20, item_y))
    
    def draw_files(self, screen, x, y, width, height):
        """Draw the file list with scrolling capability."""
        pygame.draw.rect(screen, Config.GRAY, (x, y, width, height), 2)
        screen.blit(self.font.render("Files", True, Config.BLACK), (x + 10, y + 10))
        
        # File display area
        display_y = y + 40
        display_height = height - 50
        
        # Get file list (either search results or all files)
        if self.search_mode and self.search_results:
            file_list = [(fname, self.filesystem.files[fname]) for fname in self.search_results]
        else:
            file_list = list(self.filesystem.files.items())
        
        # Calculate scrolling parameters
        max_display = Config.MAX_DISPLAY_FILES
        scroll_max = max(0, len(file_list) - max_display)
        self.file_scroll = max(0, min(self.file_scroll, scroll_max))
        
        # Draw files
        mouse_x, mouse_y = pygame.mouse.get_pos()
        for i, (fname, (content, size, path)) in enumerate(file_list[self.file_scroll:self.file_scroll + max_display]):
            file_y = display_y + i * 30
            
            # Highlight selected or hovered file
            color = Config.LIGHT_BLUE if fname == self.selected_file else Config.WHITE
            if x <= mouse_x <= x + width - 20 and file_y <= mouse_y <= file_y + 30:
                color = Config.DARK_GRAY
            
            pygame.draw.rect(screen, color, (x, file_y, width - 20, 30))
            text = self.font.render(f"{fname} ({size}KB)", True, Config.BLACK)
            screen.blit(text, (x + 10, file_y + 5))
        
        # Draw scrollbar if needed
        if len(file_list) > max_display:
            scrollbar_x = x + width - 20
            scrollbar_y = display_y
            scrollbar_height = display_height
            
            # Draw scrollbar background
            pygame.draw.rect(screen, Config.GRAY, (scrollbar_x, scrollbar_y, 20, scrollbar_height), 2)
            
            # Draw scrollbar handle
            handle_height = max(30, scrollbar_height * max_display / len(file_list))
            handle_y = scrollbar_y + (self.file_scroll / scroll_max) * (scrollbar_height - handle_height)
            pygame.draw.rect(screen, Config.DARK_GRAY, (scrollbar_x, handle_y, 20, handle_height))
    
    def draw_dashboard(self, screen, x, y, width, height):
        """Draw the dashboard with system metrics."""
        pygame.draw.rect(screen, Config.GRAY, (x, y, width, height), 2)
        screen.blit(self.font.render("Dashboard", True, Config.BLACK), (x + 10, y + 10))
        
        # Basic metrics
        metrics = [
            f"Active Processes: {len(self.scheduler.ready_queue) + (1 if self.scheduler.current_process else 0)}",
            f"IO Processes: {len(self.scheduler.io_queue)}",
            f"Terminated: {len(self.scheduler.terminated)}",
            f"Files: {len(self.filesystem.files)}",
            f"Current Time: {self.scheduler.current_time}",
            f"Scheduler: {self.scheduler.algorithm}"
        ]
        
        for i, metric in enumerate(metrics):
            text = self.font.render(metric, True, Config.BLACK)
            screen.blit(text, (x + 10, y + 40 + i * 30))
    
    def draw_status_bar(self, screen, x, y, width, height):
        """Draw the status bar with current message."""
        pygame.draw.rect(screen, Config.DARK_GRAY, (x, y, width, height))
        text = self.font.render(self.status_message, True, Config.WHITE)
        screen.blit(text, (x + 10 , y + (height - text.get_height()) // 2))
    
    def draw_buttons(self, screen):
        """Draw all UI buttons."""
        mouse_pos = pygame.mouse.get_pos()
        
        for button in self.buttons:
            button.draw(screen, self.font, mouse_pos)
            
            # Show tooltip if mouse is over button
            if button.rect.collidepoint(mouse_pos):
                tooltip = self.tooltips.get(button.text)
                if tooltip:
                    tooltip.show(mouse_pos)
                    self.active_tooltip = tooltip
    
    def draw_gantt_chart(self, screen, x, y, width, height):
        pygame.draw.rect(screen, Config.GRAY, (x, y, width, height), 2)
        screen.blit(self.font.render("Gantt Chart", True, Config.BLACK), (x + 10, y + 10))

        if not self.execution_history:
            return

        # Calculate time scale
        time_range = self.scheduler.current_time if self.scheduler.current_time > 0 else 1
        scale = (width - 20) / time_range

        bar_y = y + 40
        bar_height = 30

        for pid, start, end in self.execution_history:
            bar_x = x + 10 + int(start * scale)
            bar_width = max(1, int((end - start) * scale))
            color = [pid * 30 % 256, pid * 70 % 256, pid * 110 % 256]  # Deterministic colors
            pygame.draw.rect(screen, color, (bar_x, bar_y, bar_width, bar_height))
            pid_text = self.font.render(str(pid), True, Config.BLACK)
            screen.blit(pid_text, (bar_x + 2, bar_y + 5))

    def draw_cpu_utilization_graph(self, screen, x, y, width, height):
        pygame.draw.rect(screen, Config.GRAY, (x, y, width, height), 2)
        screen.blit(self.font.render("CPU Utilization", True, Config.BLACK), (x + 10, y + 5))
    
        if not self.utilization_history:
            return

        graph_bottom = y + height - 10
        max_height = height - 30
        bar_width = width / self.utilization_history_length

        for i, util in enumerate(self.utilization_history):
            bar_x = x + int(i * bar_width)
            bar_h = int(util * max_height)
            pygame.draw.line(
                screen,
                Config.GREEN,
                (bar_x, graph_bottom),
                (bar_x, graph_bottom - bar_h),
                2
            )

    def draw_directory_structure(self, screen, x, y, width, height):
        """Draw the directory structure visualization."""
        pygame.draw.rect(screen, Config.GRAY, (x, y, width, height), 2)
        screen.blit(self.font.render("Directory Structure", True, Config.BLACK), (x + 10, y + 10))
        
        # Draw current directory path
        path_text = self.font.render(f"Current: {self.filesystem.current_dir}", True, Config.BLACK)
        screen.blit(path_text, (x + 10, y + 30))
        
        # Draw directory tree
        tree_x = x + 10
        tree_y = y + 60
        tree_width = width - 20
        tree_height = height - 80
        
        # Draw tree background
        pygame.draw.rect(screen, Config.WHITE, (tree_x, tree_y, tree_width, tree_height))
        
        # Calculate visible items
        items_per_page = tree_height // 25
        total_items = len(self.filesystem.list_contents())
        max_scroll = max(0, total_items - items_per_page)
        self.directory_scroll = min(self.directory_scroll, max_scroll)
        
        # Draw directory contents
        mouse_x, mouse_y = pygame.mouse.get_pos()
        for i, (name, type_) in enumerate(self.filesystem.list_contents()[self.directory_scroll:self.directory_scroll + items_per_page]):
            item_y = tree_y + i * 25
            
            # Highlight selected or hovered item
            color = Config.LIGHT_BLUE if (name, type_) == self.selected_directory_item else Config.WHITE
            if tree_x <= mouse_x <= tree_x + tree_width and item_y <= mouse_y <= item_y + 25:
                color = Config.DARK_GRAY
            
            pygame.draw.rect(screen, color, (tree_x, item_y, tree_width, 25))
            
            # Draw icon based on type
            if type_ == "dir":
                pygame.draw.rect(screen, (255, 165, 0), (tree_x + 5, item_y + 5, 15, 15))  # Orange square for directory
            else:
                pygame.draw.rect(screen, (0, 150, 255), (tree_x + 5, item_y + 5, 15, 15))  # Blue square for file
            
            # Draw name
            name_text = self.font.render(name, True, Config.BLACK)
            screen.blit(name_text, (tree_x + 25, item_y + 5))
        
        # Draw scrollbar if needed
        if total_items > items_per_page:
            scrollbar_x = tree_x + tree_width - 20
            scrollbar_height = tree_height
            handle_height = max(30, scrollbar_height * items_per_page / total_items)
            handle_y = tree_y + (self.directory_scroll / max(1, max_scroll)) * (scrollbar_height - handle_height)
            
            # Draw scrollbar background
            pygame.draw.rect(screen, Config.GRAY, (scrollbar_x, tree_y, 20, scrollbar_height), 2)
            
            # Draw scrollbar handle
            pygame.draw.rect(screen, Config.DARK_GRAY, (scrollbar_x, handle_y, 20, handle_height))
            
            # Draw scroll arrows
            arrow_size = 10
            # Up arrow
            pygame.draw.polygon(screen, Config.BLACK, [
                (scrollbar_x + 10, tree_y + arrow_size),
                (scrollbar_x, tree_y + arrow_size * 2),
                (scrollbar_x + 20, tree_y + arrow_size * 2)
            ])
            # Down arrow
            pygame.draw.polygon(screen, Config.BLACK, [
                (scrollbar_x + 10, tree_y + scrollbar_height - arrow_size),
                (scrollbar_x, tree_y + scrollbar_height - arrow_size * 2),
                (scrollbar_x + 20, tree_y + scrollbar_height - arrow_size * 2)
            ])
            
            # Store scrollbar info for dragging
            self.scrollbar_info = {
                'x': scrollbar_x,
                'y': tree_y,
                'height': scrollbar_height,
                'handle_height': handle_height,
                'max_scroll': max_scroll
            }
        else:
            self.scrollbar_info = None

    def draw(self):
        """Draw the entire UI."""
        # Clear screen
        self.screen.fill(Config.WHITE)
        
        # Draw FPS dropdown
        self.fps_dropdown.draw(self.screen, pygame.mouse.get_pos())
        
        # Draw process management section
        pygame.draw.rect(self.screen, Config.GRAY, (10, 10, 380, 250), 2)
        self.screen.blit(self.font.render(f"Processes ({self.scheduler.algorithm})", True, Config.BLACK), (20, 20))
        self.screen.blit(self.font.render(f"Time: {self.scheduler.current_time}", True, Config.BLACK), (200, 20))
        
        # Draw ready queue processes
        for i, proc in enumerate(self.scheduler.ready_queue):
            proc.y = 70 + i * 40
            self.draw_process(self.screen, proc, proc.y)
        
        # Draw running process
        if self.scheduler.current_process:
            self.scheduler.current_process.y = 70
            self.draw_process(self.screen, self.scheduler.current_process, self.scheduler.current_process.y)
        
        # Draw IO queue
        if self.scheduler.io_queue:
            self.screen.blit(self.font.render("IO Queue:", True, Config.BLACK), (20, 200))
            for i, proc in enumerate(self.scheduler.io_queue):
                text = self.font.render(f"PID: {proc.pid} (Wait: {proc.io_wait_time})", True, Config.ORANGE)
                self.screen.blit(text, (100, 200 + i * 20))
        
        # Draw terminated processes
        pygame.draw.rect(self.screen, Config.GRAY, (10, 270, 380, 150), 2)
        self.screen.blit(self.font.render("Terminated", True, Config.BLACK), (20, 280))
        
        terminated_to_show = self.scheduler.terminated[-3:] if len(self.scheduler.terminated) > 3 else self.scheduler.terminated
        for i, proc in enumerate(terminated_to_show):
            text = self.font.render(f"PID: {proc.pid} (Done at {proc.completion_time})", True, Config.BLACK)
            self.screen.blit(text, (20, 310 + i * 30))
        
        # Draw memory section
        self.draw_memory(self.screen, 410, 10, 380, 200)
        
        # Draw file system section
        self.draw_files(self.screen, 410, 220, 380, 150)
        
        # Draw storage section
        self.draw_storage(self.screen, 410, 380, 380, 200)
        
        # Draw dashboard
        self.draw_dashboard(self.screen, 800, 10, 190, 570)

        #Draw Gantt chart
        self.draw_gantt_chart(self.screen, 805, 600, 190, 80)
        
        #Draw CPU utilization chart
        self.draw_cpu_utilization_graph(self.screen, 805, 690, 190, 80)

        # Draw performance metrics
        self.draw_performance_metrics(self.screen)
        
        # Draw status bar
        self.draw_status_bar(self.screen, 10, 610, 780, 30)
        
        # Draw buttons
        self.draw_buttons(self.screen)
        
        # Draw text input if active
        if self.rename_mode or self.search_mode:
            self.text_input.draw(self.screen)
        
        # Draw active tooltip if any
        if self.active_tooltip and self.active_tooltip.active:
            self.active_tooltip.draw(self.screen)
        
        # Draw directory structure
        self.draw_directory_structure(self.screen, 10, 430, 380, 170)
        
        # Draw text editor if active
        if self.editor_active:
            save_button, close_button = self.draw_editor(self.screen)
            
            # Handle editor button clicks
            mouse_pos = pygame.mouse.get_pos()
            if pygame.mouse.get_pressed()[0]:  # Left mouse button
                if save_button.collidepoint(mouse_pos):
                    if self.filesystem.write_file(self.edited_file, self.editor_content):
                        self.status_message = f"Saved changes to {self.edited_file}"
                    else:
                        self.status_message = f"Failed to save {self.edited_file}"
                elif close_button.collidepoint(mouse_pos):
                    self.toggle_editor()
        
        # Update display
        pygame.display.flip()
    

    def handle_events(self):
        """Handle user input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            
            # Handle FPS dropdown
            if self.fps_dropdown.handle_event(event):
                Config.FPS = self.fps_dropdown.selected
                self.status_message = f"FPS set to {Config.FPS}"
            
            # Handle text input events
            if self.rename_mode or self.search_mode:
                if self.text_input.handle_event(event):
                    if self.rename_mode:
                        if self.selected_file:
                            # Extract just the filename from the full path
                            old_filename = self.selected_file.lstrip('/')
                            new_filename = self.text_input.text
                            
                            # Rename file
                            if self.filesystem.rename_file(old_filename, new_filename):
                                self.status_message = f"Renamed to {new_filename}"
                                self.selected_file = new_filename
                            else:
                                self.status_message = "Rename failed - file already exists or invalid name"
                        else:
                            # Create directory
                            if self.filesystem.create_directory(self.text_input.text):
                                self.status_message = f"Directory {self.text_input.text} created"
                            else:
                                self.status_message = "Failed to create directory"
                        self.rename_mode = False
                    elif self.search_mode:
                        # Search files
                        self.search_results = self.filesystem.search_files(self.text_input.text)
                        if self.search_results:
                            self.status_message = f"Found {len(self.search_results)} matches"
                        else:
                            self.status_message = "No files found"
            
            # Handle mouse events
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Handle button clicks
                for button in self.buttons:
                    if button.handle_event(event):
                        break
                
                # Handle file selection
                if 410 <= event.pos[0] <= 770 and 260 <= event.pos[1] <= 350:
                    for i, fname in enumerate(list(self.filesystem.files.keys())[self.file_scroll:self.file_scroll + Config.MAX_DISPLAY_FILES]):
                        file_y = 260 + i * 30
                        if file_y <= event.pos[1] <= file_y + 30:
                            self.selected_file = fname.lstrip('/')
                            self.status_message = f"Selected: {self.selected_file}"
                            if event.button == 3:  # Right click
                                self.rename_mode = True
                                self.text_input.text = self.selected_file
                                self.text_input.active = True
                
                # Handle directory item selection
                if 10 <= event.pos[0] <= 390 and 490 <= event.pos[1] <= 590:
                    items_per_page = 160 // 25
                    start_idx = self.directory_scroll
                    clicked_idx = (event.pos[1] - 490) // 25
                    
                    # Get directory contents
                    dir_contents = self.filesystem.list_contents()
                    
                    # Check if the clicked index is valid
                    if 0 <= clicked_idx < items_per_page and start_idx + clicked_idx < len(dir_contents):
                        self.selected_directory_item = dir_contents[start_idx + clicked_idx]
                        
                        # Handle double click
                        if event.button == 1:  # Left click
                            name, type_ = self.selected_directory_item
                            if type_ == "dir":
                                self.filesystem.change_directory(name)
                            else:
                                self.selected_file = name
                                if name.endswith('.txt'):
                                    self.toggle_editor()
                        
                        # Handle right click for context menu
                        elif event.button == 3:  # Right click
                            name, type_ = self.selected_directory_item
                            if type_ == "dir":
                                self.filesystem.delete_directory(name)
                            else:
                                self.filesystem.delete_file(name)
                
                # Handle scrollbar
                if 770 <= event.pos[0] <= 790 and 260 <= event.pos[1] <= 350:
                    if event.button == 4:  # Scroll up
                        self.file_scroll = max(0, self.file_scroll - 1)
                    elif event.button == 5:  # Scroll down
                        max_scroll = max(0, len(self.filesystem.files) - Config.MAX_DISPLAY_FILES)
                        self.file_scroll = min(max_scroll, self.file_scroll + 1)
                
                # Handle directory scrollbar
                if self.scrollbar_info:
                    scrollbar_x = self.scrollbar_info['x']
                    scrollbar_y = self.scrollbar_info['y']
                    scrollbar_height = self.scrollbar_info['height']
                    
                    # Check if clicked on scrollbar
                    if scrollbar_x <= event.pos[0] <= scrollbar_x + 20:
                        # Check if clicked on up arrow
                        if scrollbar_y <= event.pos[1] <= scrollbar_y + 20:
                            self.directory_scroll = max(0, self.directory_scroll - 1)
                        # Check if clicked on down arrow
                        elif scrollbar_y + scrollbar_height - 20 <= event.pos[1] <= scrollbar_y + scrollbar_height:
                            self.directory_scroll = min(self.scrollbar_info['max_scroll'], self.directory_scroll + 1)
                        # Check if clicked on track
                        else:
                            # Calculate new scroll position based on click position
                            click_pos = event.pos[1] - scrollbar_y
                            self.directory_scroll = int((click_pos / scrollbar_height) * self.scrollbar_info['max_scroll'])
            
            # Handle mouse wheel events for directory scrolling
            if event.type == pygame.MOUSEWHEEL:
                # Check if mouse is over directory area
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if 10 <= mouse_x <= 390 and 490 <= mouse_y <= 590:
                    if self.scrollbar_info:
                        if event.y > 0:  # Scroll up
                            self.directory_scroll = max(0, self.directory_scroll - 1)
                        elif event.y < 0:  # Scroll down
                            self.directory_scroll = min(self.scrollbar_info['max_scroll'], self.directory_scroll + 1)
            
            # Handle mouse motion for scrollbar dragging
            if event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0] and self.scrollbar_info:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                scrollbar_x = self.scrollbar_info['x']
                scrollbar_y = self.scrollbar_info['y']
                scrollbar_height = self.scrollbar_info['height']
                
                if scrollbar_x <= mouse_x <= scrollbar_x + 20:
                    # Calculate new scroll position based on mouse position
                    mouse_pos = max(0, min(mouse_y - scrollbar_y, scrollbar_height))
                    self.directory_scroll = int((mouse_pos / scrollbar_height) * self.scrollbar_info['max_scroll'])
            
            # Handle keyboard events for directory navigation
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.filesystem.change_directory("..")
                elif event.key == pygame.K_HOME:
                    self.filesystem.change_directory("/")
            
            # Reset tooltip when mouse moves
            if event.type == pygame.MOUSEMOTION:
                if self.active_tooltip:
                    self.active_tooltip.hide()
                    self.active_tooltip = None
            
            # Handle text editor events
            if self.editor_active:
                self.handle_editor_events(event)
                
                # Handle editor window close
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_pos = pygame.mouse.get_pos()
                    editor_x, editor_y = 200, 200
                    editor_width, editor_height = 600, 400
                    
                    # Check if click is outside editor window
                    if not (editor_x <= mouse_pos[0] <= editor_x + editor_width and 
                           editor_y <= mouse_pos[1] <= editor_y + editor_height):
                        self.toggle_editor()
    
    def run(self):
        """Main simulation loop."""
        try:
            while self.running:
                # Handle events
                self.handle_events()
                
                # Update simulation (auto-step if not paused)
                if not self.paused:
                    self.step()
                
                # Draw UI
                self.draw()
                
                # Cap the frame rate using the selected FPS
                self.clock.tick(Config.FPS)
        except Exception as e:
            logger.critical(f"Runtime error: {e}")
            raise
        finally:
            logger.info("OSSim shutting down")
            pygame.quit()

    def create_directory(self):
        """Create a new directory in the current location."""
        self.rename_mode = True
        self.text_input.active = True
        self.text_input.text = ""
        self.status_message = "Enter directory name"

    def toggle_editor(self):
        """Toggle the text editor window."""
        if not self.selected_file:
            self.status_message = "No file selected"
            return

        # Clean the filename by removing any leading slashes
        filename = self.selected_file.lstrip('/')
        
        # Check if the file is a text file
        if not filename.endswith('.txt'):
            self.status_message = "Can only edit .txt files"
            return

        # Read the file content
        content = self.filesystem.read_file(filename)
        if content is None:
            self.status_message = f"Failed to read file: {filename}"
            return

        # Store current window size
        current_size = pygame.display.get_surface().get_size()
        
        # Initialize the editor with the file content
        self.editor = TextEditor(self.font)
        self.editor.initialize()
        self.editor.open(filename, content)
        
        # Run the editor in a separate window
        self.editor.run(self.filesystem)
        
        # Clear the editor reference after it's closed
        self.editor = None
        
        # Restore main window size
        pygame.display.set_mode((Config.WIDTH, Config.HEIGHT))
        
        self.status_message = f"Closed editor for {filename}"

    def draw_editor(self, screen):
        """Draw the text editor interface."""
        if not self.editor_active:
            return
            
        # Draw editor window
        editor_x, editor_y = 200, 200
        editor_width, editor_height = 600, 400
        pygame.draw.rect(screen, Config.WHITE, (editor_x, editor_y, editor_width, editor_height))
        pygame.draw.rect(screen, Config.BLACK, (editor_x, editor_y, editor_width, editor_height), 2)
        
        # Draw title
        title = f"Editing: {self.edited_file}"
        title_surf = self.font.render(title, True, Config.BLACK)
        screen.blit(title_surf, (editor_x + 10, editor_y + 10))
        
        # Draw content area
        content_area = pygame.Rect(editor_x + 10, editor_y + 40, editor_width - 20, editor_height - 80)
        pygame.draw.rect(screen, Config.WHITE, content_area)
        pygame.draw.rect(screen, Config.GRAY, content_area, 1)
        
        # Draw text content
        lines = self.editor_content.split('\n')
        visible_lines = (content_area.height - 20) // 20  # Approximate lines that fit
        
        for i, line in enumerate(lines[self.editor_scroll:self.editor_scroll + visible_lines]):
            line_surf = self.font.render(line, True, Config.BLACK)
            screen.blit(line_surf, (content_area.x + 5, content_area.y + 5 + i * 20))
        
        # Draw cursor
        cursor_x = content_area.x + 5 + self.font.size(self.editor_content[:self.editor_cursor])[0]
        cursor_y = content_area.y + 5 + (self.editor_cursor - self.editor_scroll) * 20
        pygame.draw.line(screen, Config.BLACK, (cursor_x, cursor_y), (cursor_x, cursor_y + 15), 2)
        
        # Draw scrollbar
        if len(lines) > visible_lines:
            scrollbar_x = content_area.x + content_area.width - 20
            scrollbar_height = content_area.height
            handle_height = max(30, scrollbar_height * visible_lines / len(lines))
            handle_y = content_area.y + (self.editor_scroll / (len(lines) - visible_lines)) * (scrollbar_height - handle_height)
            
            pygame.draw.rect(screen, Config.GRAY, (scrollbar_x, content_area.y, 20, scrollbar_height), 1)
            pygame.draw.rect(screen, Config.DARK_GRAY, (scrollbar_x, handle_y, 20, handle_height))
        
        # Draw buttons
        save_button = pygame.Rect(editor_x + editor_width - 100, editor_y + editor_height - 30, 80, 25)
        close_button = pygame.Rect(editor_x + 10, editor_y + editor_height - 30, 80, 25)
        
        pygame.draw.rect(screen, Config.GREEN, save_button)
        pygame.draw.rect(screen, Config.RED, close_button)
        
        save_text = self.font.render("Save", True, Config.WHITE)
        close_text = self.font.render("Close", True, Config.WHITE)
        
        screen.blit(save_text, (save_button.x + 25, save_button.y + 5))
        screen.blit(close_text, (close_button.x + 20, close_button.y + 5))
        
        return save_button, close_button

    def handle_editor_events(self, event):
        """Handle text editor input events."""
        if not self.editor_active:
            return
            
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                if self.editor_cursor > 0:
                    self.editor_content = self.editor_content[:self.editor_cursor-1] + self.editor_content[self.editor_cursor:]
                    self.editor_cursor -= 1
            elif event.key == pygame.K_DELETE:
                if self.editor_cursor < len(self.editor_content):
                    self.editor_content = self.editor_content[:self.editor_cursor] + self.editor_content[self.editor_cursor+1:]
            elif event.key == pygame.K_LEFT:
                self.editor_cursor = max(0, self.editor_cursor - 1)
            elif event.key == pygame.K_RIGHT:
                self.editor_cursor = min(len(self.editor_content), self.editor_cursor + 1)
            elif event.key == pygame.K_UP:
                # Move cursor up one line
                current_line_start = self.editor_content.rfind('\n', 0, self.editor_cursor)
                if current_line_start != -1:
                    prev_line_start = self.editor_content.rfind('\n', 0, current_line_start)
                    if prev_line_start != -1:
                        self.editor_cursor = prev_line_start + (self.editor_cursor - current_line_start)
                    else:
                        self.editor_cursor = 0
            elif event.key == pygame.K_DOWN:
                # Move cursor down one line
                current_line_start = self.editor_content.rfind('\n', 0, self.editor_cursor)
                next_line_start = self.editor_content.find('\n', self.editor_cursor)
                if next_line_start != -1:
                    self.editor_cursor = next_line_start + 1
            elif event.key == pygame.K_RETURN:
                self.editor_content = self.editor_content[:self.editor_cursor] + '\n' + self.editor_content[self.editor_cursor:]
                self.editor_cursor += 1
            else:
                # Add regular characters
                self.editor_content = self.editor_content[:self.editor_cursor] + event.unicode + self.editor_content[self.editor_cursor:]
                self.editor_cursor += 1
            
            # Update scroll position
            lines = self.editor_content.split('\n')
            current_line = len(self.editor_content[:self.editor_cursor].split('\n')) - 1
            visible_lines = 18  # Approximate number of visible lines
            
            if current_line < self.editor_scroll:
                self.editor_scroll = current_line
            elif current_line >= self.editor_scroll + visible_lines:
                self.editor_scroll = current_line - visible_lines + 1

    def show_deadlock_visualization(self):
        """Show the deadlock visualization window."""
        if not self.deadlock_visualizer:
            # Store current window size
            current_size = self.screen.get_size()
            
            # Create and run the visualization
            self.deadlock_visualizer = DeadlockVisualizer(self.font, self.resource_manager)
            self.deadlock_visualizer.initialize()
            self.deadlock_visualizer.run()
            self.deadlock_visualizer = None
            
            # Restore main window size
            self.screen = pygame.display.set_mode(current_size)
            pygame.display.set_caption("OSSim")

class DeadlockVisualizer:
    def __init__(self, font, resource_manager):
        self.font = font
        self.resource_manager = resource_manager
        self.screen = None
        self.clock = None
        self.running = False
        self.close_button = None
        self.width = 800  # Add width attribute
        self.height = 600  # Add height attribute

    def initialize(self):
        """Initialize the visualization window."""
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Deadlock Visualization")
        self.clock = pygame.time.Clock()
        self.close_button = Button(self.width - 100, 10, 80, 30, "Close", Config.RED, self.close)
        
        # Create close button
        self.close_button = Button(700, 550, 80, 30, "Close", Config.RED, self.close)

    def close(self):
        """Close the visualization window."""
        self.running = False

    def draw_resource_graph(self):
        # Clear the screen
        self.screen.fill(Config.WHITE)
        
        # Get deadlock information
        deadlock_detected, deadlock_info = self.resource_manager.detect_deadlock()
        
        # Draw title
        title = self.font.render("Resource Allocation Graph", True, Config.BLACK)
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 20))
        
        # Draw explanation
        if deadlock_detected:
            explanation = [
                "Deadlock Cause:",
                "A circular wait condition exists where:"
            ]
            
            # Add process-specific information
            for cycle in deadlock_info:
                for pid, resource in cycle:
                    # Get what the process is holding
                    holding = []
                    for held_pid, resources in self.resource_manager.allocated.items():
                        if held_pid == pid:
                            holding.extend(resources)
                    
                    # Get what the process is waiting for
                    waiting_for = []
                    for req_pid, resources in self.resource_manager.requests.items():
                        if req_pid == pid:
                            waiting_for.extend(resources)
                    
                    # Draw process information
                    text = f"Process {pid}: Holding {holding if holding else 'nothing'}, Waiting for {waiting_for if waiting_for else 'nothing'}"
                    explanation.append(text)
            
            # Draw explanation text
            y_offset = 60
            for line in explanation:
                text = self.font.render(line, True, Config.BLACK)
                self.screen.blit(text, (20, y_offset))
                y_offset += 30
        
        # Calculate graph dimensions and center position
        graph_width = 600
        graph_height = 400
        graph_x = (self.width - graph_width) // 2
        graph_y = 200  # Start below the explanation
        
        # Draw processes and resources
        process_positions = {}
        resource_positions = {}
        
        # Get all processes and resources
        processes = set()
        resources = set()
        
        for pid, info in self.resource_manager.allocated.items():
            processes.add(pid)
            for resource in info:
                resources.add(resource)
        
        for pid, info in self.resource_manager.requests.items():
            processes.add(pid)
            for resource in info:
                resources.add(resource)
        
        # Calculate positions in a circular layout with more spacing
        num_processes = len(processes)
        num_resources = len(resources)
        radius = min(graph_width, graph_height) // 3
        
        # Position processes in a circle
        for i, pid in enumerate(processes):
            angle = 2 * math.pi * i / num_processes
            x = graph_x + graph_width // 2 + radius * math.cos(angle)
            y = graph_y + graph_height // 2 + radius * math.sin(angle)
            process_positions[pid] = (x, y)
            
            # Draw process
            pygame.draw.circle(self.screen, Config.BLUE, (int(x), int(y)), 20)
            text = self.font.render(f"P{pid}", True, Config.WHITE)
            self.screen.blit(text, (int(x) - text.get_width() // 2, int(y) - text.get_height() // 2))
        
        # Position resources in an inner circle with more spacing
        for i, resource in enumerate(resources):
            angle = 2 * math.pi * i / num_resources
            x = graph_x + graph_width // 2 + (radius * 0.7) * math.cos(angle)
            y = graph_y + graph_height // 2 + (radius * 0.7) * math.sin(angle)
            resource_positions[resource] = (x, y)
            
            # Draw resource
            pygame.draw.rect(self.screen, Config.GREEN, (int(x) - 25, int(y) - 15, 50, 30))
            text = self.font.render(resource, True, Config.WHITE)
            self.screen.blit(text, (int(x) - text.get_width() // 2, int(y) - text.get_height() // 2))
        
        # Draw allocation edges with curves
        for pid, resources in self.resource_manager.allocated.items():
            if pid in process_positions:
                for resource in resources:
                    if resource in resource_positions:
                        start = process_positions[pid]
                        end = resource_positions[resource]
                        
                        # Calculate control points for the curve
                        mid_x = (start[0] + end[0]) / 2
                        mid_y = (start[1] + end[1]) / 2
                        control_x = mid_x + (end[1] - start[1]) * 0.2
                        control_y = mid_y - (end[0] - start[0]) * 0.2
                        
                        # Draw curved line
                        points = []
                        for t in range(0, 101, 5):
                            t = t / 100
                            x = (1-t)**2 * start[0] + 2*(1-t)*t * control_x + t**2 * end[0]
                            y = (1-t)**2 * start[1] + 2*(1-t)*t * control_y + t**2 * end[1]
                            points.append((int(x), int(y)))
                        
                        for i in range(len(points)-1):
                            pygame.draw.line(self.screen, Config.BLACK, points[i], points[i+1], 2)
                        
                        # Draw arrow
                        arrow_size = 10
                        angle = math.atan2(end[1] - points[-2][1], end[0] - points[-2][0])
                        arrow_points = [
                            (end[0] - arrow_size * math.cos(angle - math.pi/6),
                             end[1] - arrow_size * math.sin(angle - math.pi/6)),
                            end,
                            (end[0] - arrow_size * math.cos(angle + math.pi/6),
                             end[1] - arrow_size * math.sin(angle + math.pi/6))
                        ]
                        pygame.draw.polygon(self.screen, Config.BLACK, arrow_points)
        
        # Draw request edges with curves
        for pid, resources in self.resource_manager.requests.items():
            if pid in process_positions:
                for resource in resources:
                    if resource in resource_positions:
                        start = process_positions[pid]
                        end = resource_positions[resource]
                        
                        # Calculate control points for the curve
                        mid_x = (start[0] + end[0]) / 2
                        mid_y = (start[1] + end[1]) / 2
                        control_x = mid_x - (end[1] - start[1]) * 0.2
                        control_y = mid_y + (end[0] - start[0]) * 0.2
                        
                        # Draw curved dashed line
                        points = []
                        for t in range(0, 101, 5):
                            t = t / 100
                            x = (1-t)**2 * start[0] + 2*(1-t)*t * control_x + t**2 * end[0]
                            y = (1-t)**2 * start[1] + 2*(1-t)*t * control_y + t**2 * end[1]
                            points.append((int(x), int(y)))
                        
                        dash_length = 10
                        gap_length = 5
                        for i in range(0, len(points)-1, 2):
                            if i + 1 < len(points):
                                pygame.draw.line(self.screen, Config.RED, points[i], points[i+1], 2)
                        
                        # Draw arrow
                        arrow_size = 10
                        angle = math.atan2(end[1] - points[-2][1], end[0] - points[-2][0])
                        arrow_points = [
                            (end[0] - arrow_size * math.cos(angle - math.pi/6),
                             end[1] - arrow_size * math.sin(angle - math.pi/6)),
                            end,
                            (end[0] - arrow_size * math.cos(angle + math.pi/6),
                             end[1] - arrow_size * math.sin(angle + math.pi/6))
                        ]
                        pygame.draw.polygon(self.screen, Config.RED, arrow_points)
        
        # Draw legend
        legend_y = graph_y + graph_height + 20
        legend_items = [
            ("Process", Config.BLUE, "circle"),
            ("Resource", Config.GREEN, "rectangle"),
            ("Allocation", Config.BLACK, "line"),
            ("Request", Config.RED, "dashed")
        ]
        
        for i, (text, color, shape) in enumerate(legend_items):
            x = 20 + i * 200
            if shape == "circle":
                pygame.draw.circle(self.screen, color, (x + 10, legend_y + 10), 10)
            elif shape == "rectangle":
                pygame.draw.rect(self.screen, color, (x, legend_y, 20, 20))
            elif shape == "line":
                pygame.draw.line(self.screen, color, (x, legend_y + 10), (x + 20, legend_y + 10), 2)
            elif shape == "dashed":
                for j in range(0, 20, 5):
                    pygame.draw.line(self.screen, color, (x + j, legend_y + 10), (x + j + 3, legend_y + 10), 2)
            
            text_surface = self.font.render(text, True, Config.BLACK)
            self.screen.blit(text_surface, (x + 30, legend_y))
        
        # Draw close button
        self.close_button.draw(self.screen, self.font, pygame.mouse.get_pos())
        
        pygame.display.flip()

    def handle_events(self):
        """Handle visualization window events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if self.close_button.handle_event(event):
                    self.close()
                    return

    def run(self):
        """Run the visualization main loop."""
        self.running = True
        while self.running:
            self.handle_events()
            self.draw_resource_graph()
            self.clock.tick(60)

# Run the simulator
if __name__ == "__main__":
    try:
        sim = OSSim()
        sim.run()
    except Exception as e:
        logging.critical(f"Fatal error: {e}")
        pygame.quit()
        raise
