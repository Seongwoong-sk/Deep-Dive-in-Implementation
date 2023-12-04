import torch
import torch.nn as nn   # Basic building blocks for graphs
from models.common import *
import torch.nn.functional as F   # Activation functions & loss functions
import torch.optim as optim   # optimization algorithms (e.g. optimizer)


class YOLOv7(nn.Module):  # NOTE: nn.Module is Base class for all neural network modules. Your models should also subclass this class.
    def __init__(self):
        super().__init__()  # 부모 클래스(super)인 nn.Module의 생성자를 불러주는 것을 의미

        '''Input Tensor Size = (n, 3, 640, 640)'''

        ''' BACKBONE '''
        self.backbone = Backbone() # Github line 0 ~ 50
        
        '''NECK 1 (TOP-DOWN)'''
        self.neck1 = Neck1() # Github line 51 ~75
        
        '''NECK 2 (BOTTOM-UP)'''
        self.neck2 = Neck2() # Github line 76 ~101
        
        ''' HEAD '''
        self.head = Head(nc=80) # Github line 102 ~ 105
        
    def forward(self, x):
        '''
        ✨ BACKBONE ✨
        return : x, bbone_skip1, bbone_skip2
            - bbone_skip1 output :: (n, 512, 80, 80) 
                (Neck1의 2번째 skip connection)
            - bbone_skip2 output :: (n, 1024, 40, 40)
                (Neck1의 1번째 skip connection)
        '''
        out, bbone_skip1, bbone_skip2 = self.backbone(x) 
        
        
        '''
        ✨ NECK1 (TOP-DONW) ✨
        Args (forward function)
            - bbone_skip1 :: (n, 512, 80, 80) --> Neck1의 2번째 concat
            - bbone_skip2 :: (n, 1024, 40, 40) --> Neck1의 1번째 concat
        Return
            - x
            - neck1_skip1 :: (n, 512, 20, 20) --> Neck2의 2번째 concat
            - neck1_skip2 :: (n, 256, 40, 40) --> Neck2의 1번째 concat
        '''
        neck1_out, neck1_skip1, neck1_skip2 = self.neck1(out, bbone_skip1, bbone_skip2)
        
        
        '''
        ✨ NECK2 (BOTTOM-UP) ✨
        Args (forward function)
            -  neck1_skip1 :: (n, 512, 20, 20) --> Neck2의 2번째 concat
            -  neck1_skip2 :: (n, 256, 40, 40) --> Neck2의 1번째 concat
        
        Return
            -  neck2_rep1 :: (n, 256, 40, 40) --> For RepConv output :: (n,3,40,40,85) 
            -  neck2_rep2 :: (n, 512, 20, 20) -->  For RepConv output :: (n,3,20,20,85)
        '''
        neck2_rep1, neck2_rep2 = self.neck2(neck1_out,neck1_skip1, neck1_skip2)


        '''
        ✨ HEAD ✨
        Args (forward function)
            -  neck1_out  :: (n, 128, 80, 80) --> For RepConv output :: (n,3,80,80,85)
            -  neck2_rep1 :: (n, 256, 40, 40) --> For RepConv output :: (n,3,40,40,85)
            -  neck2_rep1 :: (n, 512, 20, 20) --> For RepConv output :: (n,3,20,20,85)
        
        Return
            -  a list consisting of 3 torch.Tensors
                - [
                    torch.size([1, 3, 80, 80, 85]),
                    torch.Size([1, 3, 40, 40, 85]), 
                    torch.Size([1, 3, 20, 20, 85]), 
                   ]
                   
            - 검출기의 출력이 최종 prediction이 되도록 변환
                - 결과로 생긴 예측 bw, bh는 이미지의 높이와 넓이로 normalize됨 (training label은 이 방법으로 선택됨)
                - 따라서 만약 박스에 대한 prediction bx,by가 (0.3, 0.8)이면 13x13 피쳐맵에서 실제 높이와 넓이는 (13 x 0.3, 13 x 0.8)
        '''
        return self.head(neck1_out, neck2_rep1, neck2_rep2)