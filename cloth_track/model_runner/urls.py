from django.urls import path
from . import views

urlpatterns = [
    path('run-sam2/', views.RunSam2View.as_view(), name='run_sam2'),
    path('run-yolo/', views.RunYOLOView.as_view(), name='run_yolo'),
    path('test/', views.DoubleNumberView.as_view(), name='double_number'),
]
