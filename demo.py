import torch



#model = torch.hub.load('.', 'custom', path='./best.pt', source='local')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt',force_reload=True) 


demo= model('im2.jpg')

demo.help()
