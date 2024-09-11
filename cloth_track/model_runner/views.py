from django.http import JsonResponse
from django.views import View
from asgiref.sync import async_to_sync, sync_to_async
from threading import Thread
import asyncio
import hashlib

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

class RunCreateLipSyncView(View):
    def get(self, request):
        url = request.GET.get('url')
        if not url:
            return JsonResponse({'error': 'No URL provided'}, status=400)
        
        process_status, created = ProcessStatus.objects.get_or_create(input_url=url,model_name="CAT_LIPSYNC")
        
        if created:
            thread = Thread(target=self.run_create_lipsync_in_thread, args=(url,))
            thread.start()
            
            return JsonResponse({'message': 'Create Lipsync run initiated'})
        else:
            return JsonResponse({'message': str(process_status)})

    def run_create_lipsync_in_thread(self, url):
        cat_lipsync_runner(url)

class RunCoverFingerView(View):
    def get(self, request):
        url = request.GET.get('url')
        if not url:
            return JsonResponse({'error': 'No URL provided'}, status=400)
        
        process_status, created = ProcessStatus.objects.get_or_create(input_url=url,model_name="COVER_FINGER")
        
        if created:
            thread = Thread(target=self.run_cover_finger_in_thread, args=(url,))
            thread.start()
            
            return JsonResponse({'message': 'Cover Finger run initiated'})
        else:
            return JsonResponse({'message': str(process_status)})

    def run_cover_finger_in_thread(self, url):
        cover_finger_runner(url)

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

@method_decorator(csrf_exempt, name='dispatch')
class RunCoverFingerStringBasedView(View):
    def post(self, request):
        base64_image = request.POST.get('base64_image')
        if not base64_image:
            return JsonResponse({'error': 'No base64 image provided'}, status=400)

        # Generate a hash of the base64_image
        image_hash = hashlib.sha256(base64_image.encode()).hexdigest()[:10]
        
        process_status, created = ProcessStatus.objects.get_or_create(input_url=image_hash, model_name="COVER_FINGER_STRING_BASED")
        
        if created:
            thread = Thread(target=self.run_cover_finger_string_based_in_thread, args=(base64_image,))
            thread.start()
            
            return JsonResponse({'message': 'Cover Finger String Based run initiated'})
        else:
            return JsonResponse({
                'status': str(process_status.message),
                'completion_percentage': str(process_status.get_completion_percentage()),
                'base64_output': process_status.get_base64_output()
            })

    def run_cover_finger_string_based_in_thread(self, base64_image):
        result_base64 = cover_finger_string_based_runner(base64_image)
        process_status = ProcessStatus.objects.get(input_url=image_hash, model_name="COVER_FINGER_STRING_BASED")
        process_status.base64_output = result_base64
        process_status.save()


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

