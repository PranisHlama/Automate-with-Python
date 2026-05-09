import os
import sys
from datetime import datetime

class ToDoApp:
    def __init__(self, filename="tasks.txt"):
        self.filename=filename
        self.tasks=[]
        self.load_tasks()
    
    def load_tasks(self):
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r') as file:
                    self.tasks = [line.strip() for line in file.readlines() if line.strip()]
                    print(f"Loaded {len(self.tasks)} tasks from {self.filename}")
            
            else:
                print(f" No existing tasks file found. Starting Fresh!")
        except Exception as e:
            print(f"Error loading tasks: {e}")
            self.tasks = []

    def save_tasks(self):
        try:
            with open(self.filename, 'w') as file:
                for task in self.tasks:
                    file.write(task + '\n')
            print(f"Tasks saved to {self.filename}")
        except Exception as e:
            print(f"Error saving tasks: {e}")

    def add_task(self, task):
        if task.strip():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            formatted_task = f"[{timestamp}] {task.strip()}"
            self.tasks.append(formatted_task)
            self.save_tasks()
            print(f"Task added: {task}")
        else:
            print("Task cannot be empty!")

    def view_tasks(self):
        if not self.tasks:
            print("No tasks found!")
            return
        print("\n Your To-Do-List:")
        print("-" * 50)
        for i, task in enumerate(self.tasks, 1):
            print(f"{i}. {task}")
        print("-" * 50)
        print(f"Total Tasks: {len(self.tasks)}")

    def remove_task(self, task_number):
        try:
            if 1 <= task_number <= len(self.tasks):
                removed_task = self.tasks.pop(task_number - 1)
                self.save_tasks()
                print(f"Task completed and removed: {removed_task}")
            else:
                print(f"Invalid task number! Please choose between 1 and {len(self.tasks)}")
        except ValueError:
            print("Please enter a valid number!")

    def clear_all_tasks(self):
            if not self.tasks:
                print("No tasks to clear!")
                return
        
            confirm = input(f"Are you sure you want to delete all {len(self.tasks)} tasks? (y/N)")
            if confirm.lower() in ['y', 'yes']:
                self.tasks.clear()
                self.save_tasks()
                print("All tasks cleared!")
            else:
                print("Operation Cancelled!")

    def display_menu(self):
            print("\n To-Do List CLI App")
            print("=" * 30)
            print("1. View All Tasks")
            print("2. Add a new Tasks")
            print("3. Complete a Tasks")
            print("4. Clear all Tasks")
            print("5. Exit")
            print("=" * 30)
    
    def run(self):
        print("Welcome to your personal To-Do List Manager!")

        while True:
            self.display_menu()

            try:
                choice = input("Enter your choice (1-5): ").strip()

                if choice == '1':
                    self.view_tasks()
                
                elif choice == '2':
                    task = input("Enter your new task:").strip()
                    self.add_task(task)

                elif choice == '3':
                    self.view_tasks()
                    if self.tasks:
                        try:
                            task_num = int(input("Enter task number to complete: "))
                            self.remove_task(task_num)
                        except ValueError:
                            print("Please enter a valid number!")
                    
                elif choice == '4':
                    self.clear_all_tasks()
                
                elif choice == '5':
                    print("Thanks for using To-Do List CLI! Stay organized!")
                    sys.exit(0)
                
                else:
                    print("Invalid choice! Please select 1-5.")
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Your tasks are safely saved.")
                sys.exit(0)
            except Exception as e:
                print(f"An error occurred: {e}")

            input("\nPress Enter to continue...")

if __name__ == "__main__":
    app = ToDoApp()
    app.run()