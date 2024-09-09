from django.db import models

class ProcessStatus(models.Model):
    input_url = models.URLField(max_length=255)
    model_name = models.CharField(max_length=100)  # Add model_name field
    percentage_completion = models.DecimalField(max_digits=5, decimal_places=2, default=0.0)
    output_url = models.URLField(max_length=255, blank=True, null=True)
    base64_output = models.TextField(blank=True, null=True)
    message = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"ProcessStatus for {self.input_url} ({self.model_name}) : {self.percentage_completion}% : {self.message} : {self.output_url} : {self.base64_output}"

    def get_base64_output(self):
        """
        Returns the base64 output image if available, otherwise returns None.
        """
        if self.base64_output:
            return self.base64_output
        return None

    def get_completion_percentage(self):
        """
        Returns the completion percentage as a float.
        """
        return float(self.percentage_completion)

    class Meta:
        verbose_name = "Process Status"
        verbose_name_plural = "Process Statuses"
        unique_together = ('input_url', 'model_name') 