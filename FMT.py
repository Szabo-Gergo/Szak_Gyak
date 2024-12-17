import cv2
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from PIL import Image, ImageTk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import time


class MotionAnalysis:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.current_algo_used = "Background Subtraction MOG2"

        # Parameters for motion analysis
        self.back_sub_mog = cv2.createBackgroundSubtractorMOG2()
        self.back_sub_knn = cv2.createBackgroundSubtractorKNN()
        self.movement_area_sum = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.movement_per_second = []


        # GUI base window
        self.root = tk.Tk()
        self.root.title("Motion Analysis")
        self.root.geometry("1000x1400")

        # GUI menubar
        self.menu = tk.Menu(self.root)
        self.root.config(menu=self.menu)

        self.import_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=self.import_menu)
        self.import_menu.add_command(label="Import Video", command=self.import_video)

        # GUI Title label
        self.title_label = tk.Label(self.root, text="Motion Analysis", font=("Arial", 20))
        self.title_label.grid(column=0, row=0)

        # GUI Dropbox label
        self.algo_label = tk.Label(self.root, text="Analysis algorithms", font=("Arial", 15))
        self.algo_label.grid(column=0, row=1, pady=15)

        # GUI Dropbox
        analysis_algos = [
            "Background Subtraction MOG2",
            "Background Subtraction KNN",
            "Absolute Differencing",
        ]

        self.algo_combo = ttk.Combobox(self.root, values=analysis_algos, width=50)
        self.algo_combo.current(0)
        self.algo_combo.bind("<<ComboboxSelected>>", self.analysis_algo_set)
        self.algo_combo.grid(column=0, row=2, pady=15)

        # GUI for the given frame
        self.base_img_label = tk.Label(self.root, width=400, height=400, bd="5", relief="solid")
        self.base_img_label.grid(column=0, row=3)

        # GUI for the transformed frame
        self.transformed_img_label = tk.Label(self.root, width=400, height=400, bd="5", relief="solid")
        self.transformed_img_label.grid(column=1, row=3, padx=50)

        # Plot setup embedded in Tkinter
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.line, = self.ax.plot([], [], marker='o')
        self.ax.set_xlim(0, 60)  # X-axis for 60 seconds
        self.ax.set_ylim(0, 100)  # Y-axis for percentage
        self.ax.set_xlabel('Time (Seconds)')
        self.ax.set_ylabel('Movement Percentage (%)')
        self.ax.set_title('Percentage of Frame Area Moving (Last Minute)')

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(column=0, row=4, pady=20)


        self.average_passed = False
        self.average_passed_textbox = scrolledtext.ScrolledText(self.root, width=50,height=25, state=tk.DISABLED)
        self.average_passed_textbox.grid(column=1, row=4, pady=10, padx=10)

        # Start camera feed
        self.get_camera()
        self.root.mainloop()

    def get_camera(self):
        ret, frame = self.cap.read()

        if ret:
            # Resize the base frame to fit the label dimensions
            resized_frame = self.resize_frame(frame, 400, 400)
            imgtk = self.convert_image_to_imagetk(resized_frame)

            # Update the label with the new image
            self.base_img_label.imgtk = imgtk
            self.base_img_label.configure(image=imgtk)

            # Perform motion analysis
            transformed_imgtk = None
            match self.current_algo_used:
                case "Background Subtraction MOG2":
                    transformed_imgtk, movement_area = self.background_subtraction_with_mog2(frame)
                case "Background Subtraction KNN":
                    transformed_imgtk, movement_area = self.background_subtraction_with_knn(frame)
                case "Absolute Differencing":
                    transformed_imgtk, movement_area = self.absolute_difference(frame)

            self.transformed_img_label.imgtk = transformed_imgtk
            self.transformed_img_label.configure(image=transformed_imgtk)

            # Update movement tracking and plot
            frame_area = frame.shape[0] * frame.shape[1]
            movement_percentage = (movement_area / frame_area) * 100
            self.update_movement_data(movement_percentage)
            self.update_plot()

        # Call this function again after 30 milliseconds
        self.root.after(30, self.get_camera)

    def analysis_algo_set(self, e):
        self.current_algo_used = self.algo_combo.get()
        self.start_time = time.time()
        self.movement_area_sum = 0
        self.frame_count = 0
        self.movement_per_second = []

    def convert_image_to_imagetk(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        return imgtk

    def background_subtraction_with_mog2(self, img):
        fg_mask = self.back_sub_mog.apply(img)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        resized_mask = self.resize_frame(fg_mask, 400, 400)
        movement_area = np.sum(resized_mask > 0)  # Count non-zero pixels

        return self.convert_image_to_imagetk(resized_mask), movement_area

    def background_subtraction_with_knn(self, img):
        fg_mask = self.back_sub_knn.apply(img)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        resized_mask = self.resize_frame(fg_mask, 400, 400)
        movement_area = np.sum(resized_mask > 0)  # Count non-zero pixels

        return self.convert_image_to_imagetk(resized_mask), movement_area

    def absolute_difference(self, current_frame):
        ret, next_frame = self.cap.read()
        if not ret:
            return self.convert_image_to_imagetk(current_frame), 0

        diff = cv2.absdiff(current_frame, next_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_diff, 50, 255, cv2.THRESH_BINARY)
        resized_thresh = self.resize_frame(thresh, 400, 400)
        movement_area = np.sum(resized_thresh > 0)  # Count non-zero pixels

        return self.convert_image_to_imagetk(resized_thresh), movement_area

    def update_movement_data(self, movement_percentage):
        self.movement_area_sum += movement_percentage
        self.frame_count += 1

        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1.0:
            average_movement = self.movement_area_sum / self.frame_count
            self.movement_per_second.append(average_movement)

            if len(self.movement_per_second) > 60:
                self.movement_per_second.pop(0)

            self.movement_area_sum = 0
            self.frame_count = 0
            self.start_time = time.time()

    def update_plot(self):
        # Update the plot with the latest data
        self.line.set_xdata(range(len(self.movement_per_second)))
        self.line.set_ydata(self.movement_per_second)

        # Calculate the average movement
        if self.movement_per_second:
            average_movement = sum(self.movement_per_second) / len(self.movement_per_second)
        else:
            average_movement = 0

        for line in self.ax.lines:
            if line.get_linestyle() == '--' and line.get_color() == 'r':
                line.remove()

        self.ax.axhline(y=average_movement, color='r', linestyle='--', label='Average Movement')

        self.update_average_textbox_passed(average_movement)

        if self.movement_per_second:
            max_movement = max(self.movement_per_second)
            self.ax.set_ylim(0, max_movement)

        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.canvas.draw()


    def import_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if video_path:
            self.cap = cv2.VideoCapture(video_path)


    def resize_frame(self, frame, target_width, target_height):
        height, width = frame.shape[:2]
        scale = min(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_frame

    def update_average_textbox_passed(self, average_movement):
        # Update average passed so we update the textbox
        if self.movement_per_second:
            current_movement = self.movement_per_second[-1]
            if current_movement > average_movement and not self.average_passed:
                self.average_passed = True
                self.average_passed_textbox.configure(state=tk.NORMAL)
                self.average_passed_textbox.insert(tk.END, f"Average passed at: {time.ctime()}\n")
                self.average_passed_textbox.configure(state=tk.DISABLED)
            elif current_movement < average_movement:
                self.average_passed = False

asd = MotionAnalysis()
