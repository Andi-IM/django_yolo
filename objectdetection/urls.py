from django.conf import settings
from django.urls import path, re_path, include
from django.views.static import serve
from rest_framework import routers

from objectdetection import views

router = routers.DefaultRouter()
router.register('users', views.UserViewSet)


urlpatterns = [
    path('api/', include(router.urls)),
    path('api_auth', views.object_detection_api, name='detect'),
    path('', views.detect_request),
    re_path(r'^detection/(?P<path>.*)$', serve, {
        'document_root': settings.MEDIA_ROOT,
    }),
]
