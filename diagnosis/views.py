from django.shortcuts import render, redirect
import pickle
import numpy as np
from .forms import DoctorConsultationForm
from django.core.mail import send_mail
from django.contrib import messages

# Load the ML model
model_path = "diagnosis/ml_models/model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

def home(request):
    return render(request, 'diagnosis/home.html')

def predict(request):
    if request.method == 'POST':
        # Get input values
        features = [
            float(request.POST['mean_radius']),
            float(request.POST['mean_texture']),
            float(request.POST['mean_perimeter']),
            float(request.POST['mean_area']),
            float(request.POST['mean_smoothness'])
        ]
        # Predict using the model
        prediction = model.predict([features])[0]
        result = "Malignant (Cancer Detected)" if prediction == 0 else "Benign (No Cancer Detected)"

        return render(request, 'diagnosis/result.html', {'result': result})
    return render(request, 'diagnosis/test.html')

def consult_doctor(request):
    if request.method == 'POST':
        form = DoctorConsultationForm(request.POST)
        if form.is_valid():
            name = form.cleaned_data['name']
            email = form.cleaned_data['email']
            message = form.cleaned_data['message']

            # Email content
            subject = "Consultation Request: Breast Cancer Diagnosis"
            body = f"Name: {name}\nEmail: {email}\n\nMessage:\n{message}"
            doctor_email = "riyamustare12@gmail.com"  # Replace with doctor's email

            # Send email
            try:
                send_mail(subject, body, email, [doctor_email])
                messages.success(request, "Your message has been sent successfully! The doctor will contact you soon.")
                return redirect('home')  # Redirect to same page or success page
            except Exception as e:
                messages.error(request, f"Failed to send message: {e}")

    else:
        form = DoctorConsultationForm()

    return render(request, 'diagnosis/consult_doctor.html', {'form': form})