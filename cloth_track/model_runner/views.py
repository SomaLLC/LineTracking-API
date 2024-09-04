from django.http import JsonResponse
from django.views import View
from asgiref.sync import async_to_sync, sync_to_async
from threading import Thread
import asyncio

from .utils import *

class RunSam2View(View):
    def get(self, request):
        url = request.GET.get('url')
        if not url:
            return JsonResponse({'error': 'No URL provided'}, status=400)
        
        process_status, created = ProcessStatus.objects.get_or_create(input_url=url,model_name="SAM-2")

        if created:
            # Run sam_2_runner asynchronously
            thread = Thread(target=self.run_sam_2_in_thread, args=(url,))
            thread.start()
            
            return JsonResponse({'message': 'SAM 2 run initiated'})
        else:
            return JsonResponse({'message': str(process_status)})

    def run_sam_2_in_thread(self, url):
        sam_2_runner(url)


class RunYOLOView(View):
    def get(self, request):
        url = request.GET.get('url')
        if not url:
            return JsonResponse({'error': 'No URL provided'}, status=400)
        
        process_status, created = ProcessStatus.objects.get_or_create(input_url=url,model_name="YOLO")

        if created:
            # Run yolo_runner asynchronously
            thread = Thread(target=self.run_yolo_in_thread, args=(url,))
            thread.start()
            
            return JsonResponse({'message': 'YOLO run initiated'})
        else:
            return JsonResponse({'message': str(process_status)})

    def run_yolo_in_thread(self, url):
        yolo_runner(url)

class RunCombinedView(View):
    def get(self, request):
        url = request.GET.get('url')
        if not url:
            return JsonResponse({'error': 'No URL provided'}, status=400)
        
        process_status, created = ProcessStatus.objects.get_or_create(input_url=url,model_name="COMBINED")
        
        if created:
            # Run combined_runner asynchronously
            thread = Thread(target=self.run_combined_in_thread, args=(url,))
            thread.start()
            
            return JsonResponse({'message': 'Combined run initiated'})
        else:
            return JsonResponse({'message': str(process_status)})

    def run_combined_in_thread(self, url):
        combined_runner(url)


class DoubleNumberView(View):
    def get(self, request):
        number_str = request.GET.get('number')
        if number_str is None:
            return JsonResponse({'error': 'No number provided'}, status=400)

        try:
            number = float(number_str)
        except ValueError:
            return JsonResponse({'error': 'Invalid number'}, status=400)

        doubled_number = number * 2
        return JsonResponse({'doubled_number': doubled_number})

