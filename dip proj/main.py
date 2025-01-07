import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import pywt
from pathlib import Path

class ModernImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Image Processing Suite")
        self.root.state('zoomed')
        
        self.colors = {
            'primary': 'black',      
            'secondary': 'white',    
            'accent': 'brown',       
            'success': 'green',     
            'warning': 'yellow',      
            'error': 'red',        
            'text_light': '#660033', 
            'text_dark': 'white',    
            'background': 'black'    
        }
        
        # Initialize variables
        self.images = {'primary': None, 'secondary': None}
        self.image_paths = {'primary': None, 'secondary': None}
        self.processed_image = None
        
        # Define operation requirements
        self.operation_requirements = {
            'Image Arithmetic': {
                'Addition': 2,
                'Subtraction': 2,
                'Multiplication': 2,
                'Division': 2,
                'Blending': 2
            },
            'Affine & Logical Operations': {
                'Translation': 1,
                'Rotation': 1,
                'Scaling': 1,
                'AND': 2,
                'OR': 2,
                'XOR': 2,
                'NOT': 1
            },
            'Image Transforms': {
                'Fourier Transform': 1,
                'Wavelet Transform': 1,
                'Hough Transform': 1
            },
            'Edge Detection': {
                'Sobel': 1,
                'Canny': 1,
                'Prewitt': 1
            },
            'Face Recognition': {
                'Detect Faces': 1,
                'Recognize Faces': 1
            }
        }
        
        self.setup_styles()
        self.create_gui()
        
    def setup_styles(self):
        """Configure custom styles for widgets"""
        style = ttk.Style()
        
        # Configure main styles
        style.configure('Primary.TFrame', background=self.colors['primary'])
        style.configure('Secondary.TFrame', background=self.colors['secondary'])
        
        # Button styles
        style.configure('Primary.TButton',
                       font=('Times New Roman', 25, 'bold'),
                       padding=0,
                       background=self.colors['accent'])
        
        style.configure('Success.TButton',
                       font=('Times New Roman', 20, 'bold'),
                       padding=0, foreground=self.colors['accent'],
                       background=self.colors['success'])
        
        # Label styles
        style.configure('Header.TLabel',
                       font=('Times New Roman', 26, 'bold'),
                       foreground=self.colors['text_light'])
        
        style.configure('Info.TLabel',
                       font=('Times New Roman', 10),
                       foreground=self.colors['text_dark'],
                       background=self.colors['background'])
        
    def create_gui(self):
        """Create the main GUI layout"""
        # Configure root grid
        self.root.configure(bg=self.colors['background'])
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        
        # Create main containers
        self.create_sidebar()
        self.create_main_content()
        
    def create_sidebar(self):
        """Create the sidebar with controls"""
        sidebar = ttk.Frame(self.root, style='Primary.TFrame')
        sidebar.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        
        # Title
        title = ttk.Label(sidebar, 
                         text="Image Processing Project",
                         style='Header.TLabel')
        title.pack(pady=2, padx=10)

        title = ttk.Label(sidebar, 
                         text="by Sadaf Shaheen, Habiba Mehmood and Agha Hammad",
                         style='Header.TLabel')
        title.pack(pady=2, padx=10)
        # Category Selection
        ttk.Label(sidebar, 
                 text="Select Category",
                 style='Header.TLabel').pack(pady=(20,5))
        
        self.category_var = tk.StringVar()
        category_combo = ttk.Combobox(sidebar, 
                                    textvariable=self.category_var,
                                    values=list(self.operation_requirements.keys()),
                                    state='readonly')
        category_combo.pack(pady=5, padx=10, fill='x')
        
        # Operation Selection
        ttk.Label(sidebar, 
                 text="Select Operation",
                 style='Header.TLabel').pack(pady=(20,5))
        
        self.operation_var = tk.StringVar()
        self.operation_combo = ttk.Combobox(sidebar,
                                          textvariable=self.operation_var,
                                          state='readonly')
        self.operation_combo.pack(pady=5, padx=10, fill='x')
        
        # Image Requirements Info
        self.requirements_label = ttk.Label(sidebar,
                                          text="",
                                          style='Info.TLabel',
                                          wraplength=200)
        self.requirements_label.pack(pady=10, padx=10)
        
        # Image Loading Buttons
        self.primary_button = ttk.Button(sidebar,
                                       text="Load Primary Image",
                                       command=lambda: self.load_image('primary'),
                                       style='Primary.TButton')
        self.primary_button.pack(pady=5, padx=10, fill='x')
        
        self.secondary_button = ttk.Button(sidebar,
                                         text="Load Secondary Image",
                                         command=lambda: self.load_image('secondary'),
                                         style='Primary.TButton')
        
        # Process Button
        self.process_button = ttk.Button(sidebar,
                                       text="Process Images",
                                       command=self.process_images,
                                       style='Success.TButton')
        self.process_button.pack(pady=20, padx=10, fill='x')
        
        # Save Button
        ttk.Button(sidebar,
                  text="Save Result",
                  command=self.save_result,
                  style='Primary.TButton').pack(pady=5, padx=10, fill='x')
        
        # Bind events
        self.category_var.trace('w', self.update_operations)
        self.operation_var.trace('w', self.update_requirements)
        
    def create_main_content(self):
        """Create the main content area"""
        main_frame = ttk.Frame(self.root, style='Secondary.TFrame')
        main_frame.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)
        
        # Create image display areas
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Welcome message
        self.welcome_label = ttk.Label(self.image_frame,
                                     text="Welcome to Image Processing Suite\n\nSelect a category and operation to begin",
                                     style='Header.TLabel')
        self.welcome_label.pack(expand=True)
        
    def update_operations(self, *args):
        """Update available operations based on selected category"""
        category = self.category_var.get()
        if category in self.operation_requirements:
            operations = list(self.operation_requirements[category].keys())
            self.operation_combo['values'] = operations
            self.operation_var.set('')  # Reset operation selection
            self.update_requirements()
        
    def update_requirements(self, *args):
        """Update interface based on operation requirements"""
        category = self.category_var.get()
        operation = self.operation_var.get()
        
        if category and operation:
            required_images = self.operation_requirements[category][operation]
            if required_images == 1:
                self.requirements_label.config(
                    text="This operation requires one image")
                self.secondary_button.pack_forget()
            else:
                self.requirements_label.config(
                    text="This operation requires two images of the same size")
                self.secondary_button.pack(pady=5, padx=10, fill='x')
                
    def load_image(self, image_type):
        """Load and validate an image"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                # Load and store image
                img = cv2.imread(file_path)
                if img is None:
                    raise ValueError("Failed to load image")
                
                # Store image and path
                self.images[image_type] = img
                self.image_paths[image_type] = file_path
                
                # Validate image sizes if both images are loaded
                if self.images['primary'] is not None and self.images['secondary'] is not None:
                    if self.images['primary'].shape != self.images['secondary'].shape:
                        messagebox.showerror("Error", 
                                           "Images must be the same size for this operation")
                        self.images[image_type] = None
                        self.image_paths[image_type] = None
                        return
                
                # Update display
                self.display_loaded_images()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                
    def display_loaded_images(self):
        """Display loaded images in the main content area"""
        # Clear previous display
        for widget in self.image_frame.winfo_children():
            widget.destroy()
        
        # Create display frame
        display_frame = ttk.Frame(self.image_frame)
        display_frame.pack(expand=True, fill='both')
        
        # Display primary image if loaded
        if self.images['primary'] is not None:
            self.display_single_image(display_frame, 
                                    self.images['primary'], 
                                    "Primary Image",
                                    0)
        
        # Display secondary image if loaded
        if self.images['secondary'] is not None:
            self.display_single_image(display_frame, 
                                    self.images['secondary'], 
                                    "Secondary Image",
                                    1)
            
    def display_single_image(self, parent, image, title, column):
        """Display a single image with title"""
        # Create frame for image
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=column, padx=10, pady=10)
        
        # Convert and resize image for display
        display_size = (400, 400)  # Maximum display size
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_pil.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage and display
        photo = ImageTk.PhotoImage(image_pil)
        label = ttk.Label(frame, image=photo)
        label.image = photo  # Keep a reference
        label.pack()
        
        # Add title
        ttk.Label(frame, 
                 text=title,
                 style='Info.TLabel').pack()
        
    def validate_images(self):
        """Validate loaded images against operation requirements"""
        category = self.category_var.get()
        operation = self.operation_var.get()
        
        if not category or not operation:
            messagebox.showerror("Error", "Please select a category and operation")
            return False
        
        required_images = self.operation_requirements[category][operation]
        
        if required_images == 1:
            if self.images['primary'] is None:
                messagebox.showerror("Error", "Please load an image")
                return False
        else:
            if self.images['primary'] is None or self.images['secondary'] is None:
                messagebox.showerror("Error", "Please load both images")
                return False
            
            if self.images['primary'].shape != self.images['secondary'].shape:
                messagebox.showerror("Error", "Images must be the same size")
                return False
                
        return True
        
    def process_images(self):
        """Process images based on selected operation"""
        if not self.validate_images():
            return
        
        try:
            category = self.category_var.get()
            operation = self.operation_var.get()
            
            # Get primary image for processing
            img = self.images['primary'].copy()
            
            # Process based on category and operation
            if category == "Image Arithmetic":
                secondary_img = self.images['secondary']
                
                if operation == "Addition":
                    self.processed_image = cv2.add(img, secondary_img)
                elif operation == "Subtraction":
                    self.processed_image = cv2.subtract(img, secondary_img)
                elif operation == "Multiplication":
                    self.processed_image = cv2.multiply(img, secondary_img)
                elif operation == "Division":
                    # Avoid division by zero
                    secondary_img[secondary_img == 0] = 1
                    self.processed_image = cv2.divide(img, secondary_img)
                elif operation == "Blending":
                    alpha = 0.5  # You could make this adjustable
                    self.processed_image = cv2.addWeighted(img, alpha, 
                                                         secondary_img, 1-alpha, 0)
            
            elif category == "Affine & Logical Operations":
                height, width = img.shape[:2]
                
                if operation == "Translation":
                    M = np.float32([[1, 0, 50], [0, 1, 50]])  # 50 pixel shift
                    self.processed_image = cv2.warpAffine(img, M, (width, height))
                    
                elif operation == "Rotation":
                    center = (width//2, height//2)
                    M = cv2.getRotationMatrix2D(center, 45, 1.0)  # 45 degree rotation
                    self.processed_image = cv2.warpAffine(img, M, (width, height))
                    
                elif operation == "Scaling":
                    self.processed_image = cv2.resize(img, None, fx=1.5, fy=1.5)
                    
                elif operation in ["AND", "OR", "XOR"]:
                    secondary_img = self.images['secondary']
                    if operation == "AND":
                        self.processed_image = cv2.bitwise_and(img, secondary_img)
                    elif operation == "OR":
                        self.processed_image = cv2.bitwise_or(img, secondary_img)
                    elif operation == "XOR":
                        self.processed_image = cv2.bitwise_xor(img, secondary_img)
                        
                elif operation == "NOT":
                    self.processed_image = cv2.bitwise_not(img)
            
            elif category == "Image Transforms":
                if operation == "Fourier Transform":
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    f_transform = np.fft.fft2(gray)
                    f_shift = np.fft.fftshift(f_transform)
                    magnitude_spectrum = 20 * np.log(np.abs(f_shift))
                    self.processed_image = cv2.normalize(magnitude_spectrum, None, 0, 255, 
                                                       cv2.NORM_MINMAX).astype(np.uint8)
                
                elif operation == "Wavelet Transform":
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    coeffs = pywt.dwt2(gray, 'haar')
                    cA, (cH, cV, cD) = coeffs
                    
                    # Normalize coefficients for display
                    def normalize_coeff(coeff):
                        return cv2.normalize(coeff, None, 0, 255, 
                                          cv2.NORM_MINMAX).astype(np.uint8)
                    
                    # Combine coefficients into single image
                    top = np.hstack((normalize_coeff(cA), normalize_coeff(cH)))
                    bottom = np.hstack((normalize_coeff(cV), normalize_coeff(cD)))
                    self.processed_image = np.vstack((top, bottom))
                
                elif operation == "Hough Transform":
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
                    
                    result = img.copy()
                    if lines is not None:
                        for rho, theta in lines[:, 0]:
                            a = np.cos(theta)
                            b = np.sin(theta)
                            x0 = a * rho
                            y0 = b * rho
                            x1 = int(x0 + 1000*(-b))
                            y1 = int(y0 + 1000*(a))
                            x2 = int(x0 - 1000*(-b))
                            y2 = int(y0 - 1000*(a))
                            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    self.processed_image = result
            
            elif category == "Edge Detection":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                if operation == "Sobel":
                    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    magnitude = cv2.magnitude(sobelx, sobely)
                    self.processed_image = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                elif operation == "Canny":
                    self.processed_image = cv2.Canny(gray, 100, 200)
                    
                elif operation == "Prewitt":
                    kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
                    kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
                    img_prewittx = cv2.filter2D(gray, -1, kernelx)
                    img_prewitty = cv2.filter2D(gray, -1, kernely)
                    self.processed_image = cv2.addWeighted(img_prewittx, 0.5, 
                                                         img_prewitty, 0.5, 0)
            
            elif category == "Face Recognition":
                if operation == "Detect Faces":
                    face_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    )
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    result = img.copy()
                    for (x, y, w, h) in faces:
                        cv2.rectangle(result, (x, y), (x+w, y+h), 
                                    (255, 0, 0), 2)
                    self.processed_image = result
                    
                elif operation == "Recognize Faces":
                    # Similar to Detect Faces but with additional recognition logic
                    self.processed_image = self.detect_and_recognize_faces(img)
            
            # Display the processed image
            self.display_result()
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
    
    def detect_and_recognize_faces(self, image):
        """Advanced face detection with recognition markers"""
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        result = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = result[y:y+h, x:x+w]
            
            # Detect eyes in the face region
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), 
                            (0, 255, 0), 2)
                
            # Add label
            cv2.putText(result, 'Face Detected', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        return result
    
    def display_result(self):
        """Display the processed image result"""
        if self.processed_image is None:
            return
            
        # Clear previous display
        for widget in self.image_frame.winfo_children():
            widget.destroy()
            
        # Create result frame
        result_frame = ttk.Frame(self.image_frame)
        result_frame.pack(expand=True, fill='both')
        
        # Convert and display the processed image
        if len(self.processed_image.shape) == 2:  # Grayscale
            display_image = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB)
        else:
            display_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
            
        # Resize for display if needed
        display_size = (800, 600)  # Maximum display size
        image_pil = Image.fromarray(display_image)
        image_pil.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage and display
        photo = ImageTk.PhotoImage(image_pil)
        label = ttk.Label(result_frame, image=photo)
        label.image = photo  # Keep a reference
        label.pack(pady=10)
        
        # Add operation info
        ttk.Label(result_frame, 
                 text=f"Result: {self.category_var.get()} - {self.operation_var.get()}",
                 style='Info.TLabel').pack()
        
    def save_result(self):
        """Save the processed image to file"""
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No processed image to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"),
                      ("JPEG files", "*.jpg"),
                      ("All files", "*.*")]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.processed_image)
                messagebox.showinfo("Success", "Image saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")

def main():
    root = tk.Tk()
    app = ModernImageProcessor(root)
    root.mainloop()

if __name__ == "__main__":
    main()