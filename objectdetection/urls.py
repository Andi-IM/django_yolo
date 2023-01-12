from django.conf import settings
from django.template.defaulttags import url
from django.views.static import serve

urlpatterns = [
    url(r'^Object_Detection/(?P<path>.*)$', serve, {
        'document_root': settings.MEDIA_ROOT,
    }),
]
