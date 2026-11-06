from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path('inspect/', views.inspect_upload_view, name='inspect_upload'),
    path('inspections/<str:inspection_id>/', views.inspection_result_view, name='inspection_result'),
    path('inspections/<str:inspection_id>/pdf/', views.download_pdf_view, name='download_pdf'),
    path('history/', views.inspection_history_view, name='inspection_history'),
    path('analytics/', views.analytics_view, name='analytics'),
]

