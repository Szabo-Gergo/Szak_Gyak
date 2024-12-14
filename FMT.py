import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import time

#asd

class MotionAnalysis:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.current_algo_used = "Background Subtraction MOG2"

        #params for back seperation
        self.back_sub_mog = cv2.createBackgroundSubtractorMOG2()
        self.back_sub_knn = cv2.createBackgroundSubtractorKNN()
        self.movement_area_sum = 0
        self.frame_count = 0
        self.start_time = 0
        self.movement_per_second = []
        #params for optical flow
        # Get the first frame
        ret, self.frame1 = self.cap.read()

        # Convert the frame to grayscale
        self.prvs = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2GRAY)

        # Create a mask for the optical flow
        self.mask = np.zeros_like(self.frame1)

        # Define the parameters for the optical flow algorithm
        self.lk_params = dict(winSize=(15, 15),
                         maxLevel=1,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))


        #GUI base window
        self.root = tk.Tk()
        self.root.title("Motion Analysis")
        self.root.geometry("840x600")

        #GUI menubar
        self.menu = tk.Menu(self.root)
        self.root.config(menu=self.menu)

        self.import_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=self.import_menu)
        self.import_menu.add_command(label="Import Video", command=self.import_video)

        #GUI Title label
        self.title_label = tk.Label(self.root, text="Motion Analysis", font=("Arial", 20))
        self.title_label.grid(column=0, row=0)

        #GUI Dropbox label
        self.algo_label = tk.Label(self.root, text="Analysis algorithms", font=("Arial", 15))
        self.algo_label.grid(column=0, row=1, pady=15)


        #GUI Dropbox
        analysis_algos = [
            "Background Subtraction MOG2",
            "Background Subtraction KNN",
            "Absolute Differencing",
            "Sparse Optical Flow",
            "Dense Optical Flow"
        ]

        self.algo_combo = ttk.Combobox(self.root, values=analysis_algos, width=50)
        self.algo_combo.current(0)
        self.algo_combo.bind("<<ComboboxSelected>>", self.analysis_algo_set)
        self.algo_combo.grid(column=0, row=2, pady=15)

        #GUI for the given frame
        self.base_img_label = tk.Label(self.root, width=400, height=400, bd="5", relief="solid")
        self.base_img_label.grid(column=0, row=3)

        #GUI for the transformed frame
        self.transformed_img_label = tk.Label(self.root, width=400, height=400, bd="5", relief="solid")
        self.transformed_img_label.grid(column=2, row=3, padx=50)



        #Plot setup
        plt.ion()  # Turn on interactive mode
        fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], marker='o')
        self.ax.set_xlim(0, 60)  # X-axis for 60 seconds
        self.ax.set_ylim(0, 1000)  # Adjust the Y-axis limit as needed based on expected movement area
        self.ax.set_xlabel('Time (Seconds)')
        self.ax.set_ylabel('Average Movement (Area)')
        self.ax.set_title('Average Movement per Second (Last Minute)')



        self.get_camera()
        self.root.mainloop()


    #if the toggle is set to the camera return the image from the webcam with a tkImage format
    def get_camera(self):
        ret, prev_frame = self.cap.read()
        ret, frame = self.cap.read()

        if ret:
        # Converting image into a ImageTk
            imgtk = self.convert_image_to_imagetk(frame)
            transformed_imgtk = None

            # Update the label with the new image
            self.base_img_label.imgtk = imgtk
            self.base_img_label.configure(image=imgtk)

            match self.current_algo_used:
                case "Background Subtraction MOG2":
                    transformed_imgtk = self.background_subtraction_with_mog2(frame)
                case "Background Subtraction KNN":
                    transformed_imgtk = self.background_subtraction_with_knn(frame)
                case "Absolute Differencing":
                    transformed_imgtk = self.absolute_difference(frame, prev_frame)
                case "Sparse Optical Flow":
                    transformed_imgtk = self.sparse_optical_flow(frame)
                case "Dense Optical Flow":
                    transformed_imgtk = self.dense_optical_flow(frame)

            self.transformed_img_label.imgtk = transformed_imgtk
            self.transformed_img_label.configure(image=transformed_imgtk)

            # Call this function again after 10 milliseconds
        self.root.after(20, self.get_camera)

    def analysis_algo_set(self, e):
        self.current_algo_used = self.algo_combo.get()
        self.start_time = time.time()
        self.movement_area_sum = 0
        self.frame_count = 0

    def convert_image_to_imagetk(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        return imgtk

    #return the motion analysis converted image from these methods
    #Background subtraction mog2 better with static bacgrounds and knn is better with dynamic ones
    def background_subtraction_with_mog2(self, img):
        fg_mask = self.back_sub_mog.apply(img)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Find contours of the moving objects in the mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        movement_area = 0
        for contour in contours:
            movement_area += cv2.contourArea(contour)

        # Append the movement area for the current frame to the movement data list
        # Accumulate movement area and frame count
        self.movement_area_sum += movement_area
        self.frame_count += 1

        current_time = time.time()
        if current_time - self.start_time >= 1:
            # Calculate average movement per second
            average_movement = self.movement_area_sum / self.frame_count if self.frame_count > 0 else 0
            self.movement_per_second.append(average_movement)

            # Reset the accumulator for the next second
            self.update_plot()
            self.start_time = current_time
            self.movement_area_sum = 0
            self.frame_count = 0

        return self.convert_image_to_imagetk(fg_mask)

    def background_subtraction_with_knn(self, img):
        fg_mask = self.back_sub_knn.apply(img)
        return self.convert_image_to_imagetk(fg_mask)


    #Gets the absolute difference between the current and previous frames.
    def absolute_difference(self, img, prev_img):
        diff = cv2.absdiff(prev_img, img)
        return self.convert_image_to_imagetk(diff)



    def sparse_optical_flow(self, frame2):
        # Convert the frame to grayscale
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate the optical flow
        flow = cv2.calcOpticalFlowFarneback(self.prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Convert the optical flow to polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Scale the magnitude of the optical flow between 0 and 255
        mag_scaled = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the angle to hue
        ang_degrees = ang * 180 / np.pi / 2
        ang_scaled = cv2.normalize(ang_degrees, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the hue and magnitude to an RGB image
        hsv = np.zeros_like(self.frame1)
        hsv[..., 0] = ang_scaled
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.convertScaleAbs(mag_scaled)

        # Convert the HSV image to BGR
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Set the current frame as the previous frame for the next iteration
        self.prvs = next.copy()
        return self.convert_image_to_imagetk(bgr)


    #Update the plot so that it shows the amount of movement
    #if on video just show the full timeline
    #if on camera show the last minute of values
    def update_plot(self):
        pass


    #Calculate the average amount of movement and update the plot
    def set_treshold(self):
        pass


    #print the timestamp where the movement has passed the treshold
    def print_treshold_passed(self):
        pass


    #if the toggle is set to video show the video import menu
    #if there has been a video imported get the frames in a tkImage format
    def get_video(self):
        pass

    def import_video(self):
        print("Importing Video")


    def update_plot(self):
        """Update the plot with the last 60 seconds of movement data."""
        self.line.set_xdata(range(len(self.movement_per_second)))
        self.line.set_ydata(self.movement_per_second)
        self.ax.relim()  # Recalculate limits based on new data
        self.ax.autoscale_view(True, True, True)  # Rescale axes
        plt.draw()

asd = MotionAnalysis()