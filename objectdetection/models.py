from django.db import models


# Create your models here.
class Image(models.Model):
    title = models.CharField(max_length=50)
    file = models.FileField()

    def __str__(self):
        return self.title
