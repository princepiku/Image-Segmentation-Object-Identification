import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

class Window(tk.Tk):
        def __init__(self, master=None):
                tk.Tk.__init__(self)

                self.title('Image Segmentation')
                self.geometry(f"{500}x{250}")
                
                # Load the model
                self.net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb", "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

                # Store Coco Names in a list
                classesFile = "coco.names"
                self.classNames = open(classesFile).read().strip().split('\n')

                label = tk.Label(self, text='Image Segmentation', font=('Sitka Subheading', 20))
                label.place(relx=0.5, rely=0.1, anchor='center')

                self.open_button = tk.Button(self, text='Open Image', width=25, font=('Sitka Subheading', 15), command=self.open_image)
                self.open_button.place(relx=0.5, rely=0.4, anchor='center')

                self.compute_button = tk.Button(self, text='Compute', width=25, font=('Sitka Subheading', 15), command=self.compute)
                self.compute_button.place(relx=0.5, rely=0.6, anchor='center')

                self.show_button = tk.Button(self, text='Show', width=25, font=('Sitka Subheading', 15), command=self.show, state='disabled')
                self.show_button.place(relx=0.5, rely=0.8, anchor='center')
                
        # Load image
        def open_image(self):
                filename = filedialog.askopenfilename(initialdir=os.getcwd())
                global img
                img = cv2.imread(filename)

        def compute(self):
                # Create black image
                height, width, _ = img.shape
                global blank_mask
                blank_mask = np.zeros((height, width, 3), np.uint8)
                blank_mask[:] = (0, 0, 0)

                # Create blob from the image
                blob = cv2.dnn.blobFromImage(img, swapRB=True)

                # Detect objects
                self.net.setInput(blob)
                boxes, masks = self.net.forward(["detection_out_final", "detection_masks"])
                detection_count = boxes.shape[2]
                print(detection_count)
                if detection_count == 100:
                        self.show_button["state"]='active'
                count = 0
                for i in range(detection_count):
                        # Extract information from detection
                        box = boxes[0, 0, i]
                        class_id = int(box[1])
                        score = box[2]
                        # print(class_id, score)
                        if score < 0.6:
                                continue

                        # print(class_id)
                        class_name = (self.classNames[class_id])
                        # print(class_name, score)
                        x = int(box[3] * width)
                        y = int(box[4] * height)
                        x2 = int(box[5] * width)
                        y2 = int(box[6] * height)

                        roi = blank_mask[y: y2, x: x2]
                        roi_height, roi_width, _ = roi.shape
                        
                        # Get the mask
                        mask = masks[i, int(class_id)]
                        mask = cv2.resize(mask, (roi_width, roi_height))
                        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
                        # cv2.imshow("mask"+str(count), mask)
                        count+=1
                        # Find contours of the mask
                        contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        color = np.random.randint(0, 255, 3, dtype='uint8')
                        color = [int(c) for c in color]

                        # fill some color in segmented area
                        for cnt in contours:
                                cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))
                                # cv2.imshow("roi", roi)
                                
                        # Draw bounding box
                        cv2.rectangle(img, (x, y), (x2, y2), color, 2)
                        cv2.putText(img, class_name + " " + str(score), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

        def show(self):
                # alpha is the transparency of the first picture
                alpha = 1
                # beta is the transparency of the second picture
                beta = 0.8
                mask_img = cv2.addWeighted(img, alpha, blank_mask, beta, 0)
                cv2.imshow("Final Output", mask_img)

                cv2.imshow("Black image", blank_mask)
                cv2.imshow("Mask image", img)
                cv2.waitKey(0)


app = Window(tk.Tk)
app.mainloop()
