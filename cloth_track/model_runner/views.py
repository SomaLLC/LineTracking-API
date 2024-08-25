from django.http import JsonResponse
from django.views import View
import asyncio

from .utils import *

class RunSam2View(View):
    async def get(self, request):
        url = request.GET.get('url')
        if not url:
            return JsonResponse({'error': 'No URL provided'}, status=400)

        # Run sam_2_runner asynchronously
        asyncio.create_task(self.run_sam_async(url))
        
        return JsonResponse({'message': 'SAM 2 run initiated'})

    async def run_sam_async(self, url):
        # Run the sam_2_runner function asynchronously
        await asyncio.to_thread(sam_2_runner, url)


class RunYOLOView(View):
    async def get(self, request):
        url = request.GET.get('url')
        if not url:
            return JsonResponse({'error': 'No URL provided'}, status=400)

        # Run yolo_runner asynchronously
        asyncio.create_task(self.run_yolo_async(url))
        
        return JsonResponse({'message': 'YOLO run initiated'})

    async def run_yolo_async(self, url):
        # Run the yolo_runner function asynchronously
        await asyncio.to_thread(yolo_runner, url)


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

