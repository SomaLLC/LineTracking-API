from django.urls import path
from . import views

urlpatterns = [
    path('run-sam2/', views.RunSam2View.as_view(), name='run_sam2'),
    path('run-yolo/', views.RunYOLOView.as_view(), name='run_yolo'),
    path('run-combined/', views.RunCombinedView.as_view(), name='run_combined'),
    path('run-create-lipsync/', views.RunCreateLipSyncView.as_view(), name='run_create_lipsync'),
    path('run-cover-finger/', views.RunCoverFingerView.as_view(), name='run_cover_finger'),
    path('run-cover-finger-string-based/', views.RunCoverFingerStringBasedView.as_view(), name='run_cover_finger_string_based'),
    path('test/', views.DoubleNumberView.as_view(), name='double_number'),
]
