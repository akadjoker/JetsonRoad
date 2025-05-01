import cv2
import torch
import numpy as np
from torchvision import transforms
from model import UNet 
 
import time
 
 
from collections import deque

 

class AutonomousCar:
    def __init__(self ):
   
 
        
 
 
        self.model_path = "unet_trained"
        self.img_size = tuple((256, 256))
        self.threshold = 0.5
        self.ghost_timeout = 2.0
        self.min_distance = 40
        self.show_mask = False 
        self.record_output = False 
        
 
        self.point_history = deque(5)
        self.angle_history = deque(5)
        
        # Estado do fantasma (ponto de referência quando não há detecção)
        self.ghost_point = None
        self.ghost_time = 0
        
   
        self._init_model()
        
 
        self._init_video()
     
    
    def _init_model(self):
        try:
        
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Usamos: {self.device}")
            
  
            self.model = UNet().to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            
   
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
            ])
            
            print("Modelo carregado com sucesso")
        except Exception as e:
            print(f"Erro ao inicializar modelo: {e}")
            raise
    
    def _init_video(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception(f"Não foi possível abrir o vídeo")
            
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            self.center_x = self.width // 2
            self.base_y = self.height - 10
            
            print(f"Vídeo inicializado: {self.width}x{self.height} @ {self.fps}fps")
            
      
        except Exception as e:
            print(f"Erro ao inicializar vídeo: {e}")
            raise
    
    def encontrar_ponto(self, mask):
        """Encontra o ponto de interesse na máscara segmentada"""
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
            
        # Filtrar ruído: verificar densidade de pixels em áreas candidatas
        if len(xs) <  50:
            return None
            
        # Ordenar pontos por proximidade à base (mais perto tem prioridade)
        sorted_pts = sorted(zip(ys, xs), key=lambda p: abs(p[0] - self.base_y) + abs(p[1] - self.center_x))
        
        for y, x in sorted_pts:
            if self.base_y - y > self.min_distance:
                return int(x), int(y)
                
        return None
    
    def calcular_angulo_suavizado(self, point):
 
        if point:
            self.point_history.append(point)
        
        # Usar média dos últimos pontos para suavizar
        if len(self.point_history) > 0:
            avg_x = sum(p[0] for p in self.point_history) / len(self.point_history)
            avg_y = sum(p[1] for p in self.point_history) / len(self.point_history)
            smooth_point = (avg_x, avg_y)
            
            dx = smooth_point[0] - self.center_x
            dy = self.base_y - smooth_point[1]
            angle = np.arctan2(dx, dy)
            
            # Suavizar o ângulo também
            self.angle_history.append(angle)
            smooth_angle = sum(self.angle_history) / len(self.angle_history)
            
            return smooth_angle, smooth_point
        
        return 0, (self.center_x, self.base_y - 60)
    
    def atualizar_ghost_point(self, point):
        """Atualiza o ponto fantasma (referência quando não há detecção)"""
        if point:
            self.ghost_point = point
            self.ghost_time = time.time()
            return self.ghost_point, True  # Ponto real
        else:
            if self.ghost_point and (time.time() - self.ghost_time < self.ghost_timeout):
                return self.ghost_point, False  # Ponto fantasma ainda válido
            else:
                # Retornar para posição padrão após timeout
                default_point = (self.center_x, self.base_y - 60)
                return default_point, False  # Ponto padrão
    
    def processar_frame(self, frame):
 
        start_time = time.time()
        
        # Converter e preparar imagem para o modelo
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Inferência do modelo
        with torch.no_grad():
            pred = self.model(img_tensor)
            mask = (pred.squeeze().cpu().numpy() > self.threshold).astype(np.uint8) * 255
        
        # Redimensionar máscara para o tamanho original
        mask = cv2.resize(mask, (self.width, self.height))
        
        # Encontrar ponto alvo na máscara
        point = self.encontrar_ponto(mask)
        
        # Atualizar ponto fantasma (para manter referência quando não há detecção)
        target_point, is_real = self.atualizar_ghost_point(point)
        
        # Calcular ângulo suavizado
        angle, smooth_point = self.calcular_angulo_suavizado(target_point)
        
        # Calcular tempo de processamento
        process_time = time.time() - start_time
        
        return {
            'mask': mask,
            'point': point,
            'target_point': target_point,
            'is_real_point': is_real,
            'angle': angle,
            'smooth_point': smooth_point,
            'process_time': process_time
        }
    
    def renderizar_visualizacao(self, frame, result):
 
        overlay = frame.copy()
        
 
        if self.show_mask:
            mask_overlay = np.zeros_like(overlay)
            mask_overlay[result['mask'] > 0] = (0, 0, 255)  # BGR -> Vermelho
            overlay = cv2.addWeighted(overlay, 0.7, mask_overlay, 0.9, 0)

        # Desenhar ponto alvo
        if result['is_real_point']:
            # Ponto real (verde)
            cv2.circle(overlay, result['target_point'], 8, (0, 255, 0), -1)
        else:
            # Ponto fantasma (amarelo) ou padrão (azul)
            color = (0, 200, 200) if time.time() - self.ghost_time < self.ghost_timeout else (100, 100, 255)
            cv2.circle(overlay, result['target_point'], 8, color, -1)
        

        cv2.circle(overlay, (int(result['smooth_point'][0]), int(result['smooth_point'][1])), 5, (255, 0, 255), -1)
        
 
        angle = result['angle']
        arrow_x = int(self.center_x + 60 * np.sin(angle))
        arrow_y = int(self.base_y - 60 * np.cos(angle))
        cv2.arrowedLine(overlay, (self.center_x, self.base_y), (arrow_x, arrow_y), (0, 255, 255), 4)
        
 
        cv2.putText(overlay, f"Angle: {np.degrees(angle):.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        cv2.putText(overlay, f"FPS: {1.0/max(result['process_time'], 0.001):.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        cv2.putText(overlay, "Predict: " + ("Real" if result['is_real_point'] else "Ghost"), 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return overlay
    
    def enviar_comando(self, angle):
        steering_angle = np.degrees(angle)
        max_angle = 60.0 #?? OU
        steering_angle = max(min(steering_angle, max_angle), -max_angle)
        return steering_angle
    
    def executar(self):
        """Executa o loop principal de processamento de vídeo"""
        try:
            frame_count = 0
            total_time = 0
            
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                
                frame_count += 1
                
                # Processamento do frame
                result = self.processar_frame(frame)
                total_time += result['process_time']
                
                # Enviar comando para o hardware
                steering_angle = self.enviar_comando(result['angle']) / 255.0
                
                # Renderizar visualização
                output_frame = self.renderizar_visualizacao(frame, result)
                
                # Mostrar frame
                cv2.imshow("Main", output_frame)
                

                
                # Verificar tecla de saída
                key = cv2.waitKey(1)
                if key == ord("q"):
                    print("exit")
                    break
 
            
 
            
        except Exception as e:
            print(f"Erro durante execução: {e}")
        finally:
            self.cap.release()
 
            cv2.destroyAllWindows()
            print("Exit")

def main():
 
     
    car = AutonomousCar( )
 
    
 
    car.executar()

if __name__ == "__main__":
    main()
