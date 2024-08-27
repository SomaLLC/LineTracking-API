from django.db import models


class ProcessStatus(models.Model):
    input_url = models.URLField(max_length=255, unique=True)
    percentage_completion = models.DecimalField(max_digits=5, decimal_places=2, default=0.0)
    output_url = models.URLField(max_length=255, blank=True, null=True)
    message = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"ProcessStatus for {self.input_url} - {self.percentage_completion}%"

    class Meta:
        verbose_name = "Process Status"
        verbose_name_plural = "Process Statuses"