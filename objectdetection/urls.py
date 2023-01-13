from django.conf import settings
from django.urls import path, re_path
from django.views.static import serve

from objectdetection import views

urlpatterns = [
    path('api_request', views.object_detection_api, name='detect'),
    path('', views.detect_request),
    re_path(r'^detection/(?P<path>.*)$', serve, {
        'document_root': settings.MEDIA_ROOT,
    }),
]
