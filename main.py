import pygame
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
import sys

pygame.init()

# Константы
BLOCK_SIZE = 30
GRID_WIDTH = 10
GRID_HEIGHT = 20
SCREEN_WIDTH = GRID_WIDTH * BLOCK_SIZE + 400
SCREEN_HEIGHT = GRID_HEIGHT * BLOCK_SIZE + 100
FPS = 60

# Цвета
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
COLORS = [
    (0, 255, 255), (255, 255, 0), (128, 0, 128),
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 165, 0)
]

SHAPES = [
    [[1, 1, 1, 1]], [[1, 1], [1, 1]], [[0, 1, 0], [1, 1, 1]],
    [[0, 1, 1], [1, 1, 0]], [[1, 1, 0], [0, 1, 1]],
    [[1, 0, 0], [1, 1, 1]], [[0, 0, 1], [1, 1, 1]]
]

class Tetromino:
    def __init__(self, shape_id):
        self.shape = SHAPES[shape_id]
        self.color = COLORS[shape_id]
        self.x = GRID_WIDTH // 2 - len(self.shape[0]) // 2
        self.y = 0
    
    def rotate(self):
        self.shape = list(zip(*self.shape[::-1]))

class TetrisGame:
    def __init__(self):
        self.grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.current_piece = self.new_piece()
        self.game_over = False
        self.score = 0
        self.paused = False
        self.fall_speed = 500
    
    def new_piece(self):
        return Tetromino(np.random.randint(0, len(SHAPES)))
    
    def valid_move(self, piece, x, y):
        for row_idx, row in enumerate(piece.shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    new_x, new_y = x + col_idx, y + row_idx
                    if (new_x < 0 or new_x >= GRID_WIDTH or new_y >= GRID_HEIGHT or
                        (new_y >= 0 and self.grid[new_y][new_x])):
                        return False
        return True
    
    def place_piece(self):
        for row_idx, row in enumerate(self.current_piece.shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    x = self.current_piece.x + col_idx
                    y = self.current_piece.y + row_idx
                    if y >= 0:
                        self.grid[y][x] = self.current_piece.color
        self.clear_lines()
        self.current_piece = self.new_piece()
        if not self.valid_move(self.current_piece, self.current_piece.x, self.current_piece.y):
            self.game_over = True
    
    def clear_lines(self):
        lines_cleared = 0
        for row_idx in range(GRID_HEIGHT - 1, -1, -1):
            if all(self.grid[row_idx]):
                del self.grid[row_idx]
                self.grid.insert(0, [0 for _ in range(GRID_WIDTH)])
                lines_cleared += 1
        self.score += lines_cleared * 100
    
    def move_left(self):
        if self.valid_move(self.current_piece, self.current_piece.x - 1, self.current_piece.y):
            self.current_piece.x -= 1
    
    def move_right(self):
        if self.valid_move(self.current_piece, self.current_piece.x + 1, self.current_piece.y):
            self.current_piece.x += 1
    
    def move_down(self):
        if self.valid_move(self.current_piece, self.current_piece.x, self.current_piece.y + 1):
            self.current_piece.y += 1
        else:
            self.place_piece()
    
    def rotate_piece(self):
        original = self.current_piece.shape
        self.current_piece.rotate()
        if not self.valid_move(self.current_piece, self.current_piece.x, self.current_piece.y):
            self.current_piece.shape = original

class GestureController:
    def __init__(self, model_path):
        self.display_gesture = 'neutral'  # для отображения
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
        self.model = joblib.load(model_path)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        self.gesture_buffer = deque(maxlen=8)
        self.gesture_cooldown = 0
        self.cooldown_time = 15
        self.pause_cooldown = 0
        
        self.gesture_names = ['down', 'left', 'neutral', 'open_palm', 'right', 'up']
        self.current_gesture = 'neutral'
    
    def extract_landmarks(self, hand_landmarks):
        landmarks_list = []
        for lm in hand_landmarks.landmark:
            landmarks_list.extend([lm.x, lm.y, lm.z])
        return np.array(landmarks_list).reshape(1, -1)
    
    def get_gesture(self):
        ret, frame = self.cap.read()
        if not ret:
            return 'neutral', None
        
        # Обработка flipped frame для модели
        frame_flipped = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        gesture = 'neutral'
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            features = self.extract_landmarks(hand_landmarks)
            prediction = self.model.predict(features)[0]
            gesture = self.gesture_names[prediction]
            
            self.gesture_buffer.append(gesture)
            
            if len(self.gesture_buffer) >= 5:
                most_common = Counter(self.gesture_buffer).most_common(1)[0][0]
                gesture = most_common
        else:
            self.gesture_buffer.append('neutral')
            gesture = 'neutral'
        
        frame_display = cv2.flip(frame_flipped, 1)
        
        # ВРУЧНУЮ инвертируем landmarks по X для правильного отображения
        if results.multi_hand_landmarks:
            from mediapipe.framework.formats import landmark_pb2
            
            mirrored_landmarks = landmark_pb2.NormalizedLandmarkList()
            for landmark in results.multi_hand_landmarks[0].landmark:
                new_landmark = mirrored_landmarks.landmark.add()
                new_landmark.x = 1.0 - landmark.x  # Инвертируем X
                new_landmark.y = landmark.y
                new_landmark.z = landmark.z
            
            mp.solutions.drawing_utils.draw_landmarks(
                frame_display, mirrored_landmarks, self.mp_hands.HAND_CONNECTIONS
            )
        
        if gesture == 'open_palm':
            if self.pause_cooldown == 0:
                self.pause_cooldown = 30
                self.display_gesture = 'open_palm'
                return 'open_palm', frame_display
            else:
                self.pause_cooldown -= 1
                gesture = 'neutral'
        else:
            if self.pause_cooldown > 0:
                self.pause_cooldown -= 1
        
        if gesture == 'neutral':
            self.display_gesture = 'neutral'
        
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
            if gesture != 'neutral':
                return 'neutral', frame_display
        
        if gesture != 'neutral':
            self.gesture_cooldown = self.cooldown_time
            self.display_gesture = gesture
        
        return gesture, frame_display

def draw_game(screen, game, camera_frame, gesture_controller):
    screen.fill(BLACK)
    offset_x, offset_y = 50, 50
    
    pygame.draw.rect(screen, GRAY, (offset_x - 2, offset_y - 2, 
                     GRID_WIDTH * BLOCK_SIZE + 4, GRID_HEIGHT * BLOCK_SIZE + 4), 2)
    
    for row_idx, row in enumerate(game.grid):
        for col_idx, cell in enumerate(row):
            if cell:
                pygame.draw.rect(screen, cell,
                    (offset_x + col_idx * BLOCK_SIZE, offset_y + row_idx * BLOCK_SIZE,
                     BLOCK_SIZE - 1, BLOCK_SIZE - 1))
    
    if not game.game_over:
        for row_idx, row in enumerate(game.current_piece.shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(screen, game.current_piece.color,
                        (offset_x + (game.current_piece.x + col_idx) * BLOCK_SIZE,
                         offset_y + (game.current_piece.y + row_idx) * BLOCK_SIZE,
                         BLOCK_SIZE - 1, BLOCK_SIZE - 1))
    
    if camera_frame is not None:
        camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB)
        camera_frame = np.rot90(camera_frame)
        camera_frame = pygame.surfarray.make_surface(camera_frame)
        screen.blit(camera_frame, (GRID_WIDTH * BLOCK_SIZE + 80, 50))
    
    gesture_font = pygame.font.Font(None, 32)
    gesture_text = gesture_font.render(f"Gesture: {gesture_controller.display_gesture}", 
                                    True, (0, 255, 0))
    screen.blit(gesture_text, (GRID_WIDTH * BLOCK_SIZE + 80, 300))

    
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {game.score}", True, WHITE)
    screen.blit(score_text, (GRID_WIDTH * BLOCK_SIZE + 80, 340))
    
    font_small = pygame.font.Font(None, 22)
    instructions = [
        "Gestures:", "LEFT - Move left", "RIGHT - Move right",
        "UP - Rotate", "DOWN - Soft drop", "PALM - Pause", "",
        "Keyboard:", "ESC - Quit", "R - Restart"
    ]
    y_pos = 390
    for line in instructions:
        text = font_small.render(line, True, WHITE)
        screen.blit(text, (GRID_WIDTH * BLOCK_SIZE + 80, y_pos))
        y_pos += 23
    
    if game.game_over:
        font_big = pygame.font.Font(None, 72)
        text = font_big.render("GAME OVER", True, WHITE)
        rect = text.get_rect(center=(GRID_WIDTH * BLOCK_SIZE // 2 + offset_x, 
                                     GRID_HEIGHT * BLOCK_SIZE // 2))
        screen.blit(text, rect)
    
    if game.paused:
        font_big = pygame.font.Font(None, 72)
        text = font_big.render("PAUSED", True, WHITE)
        rect = text.get_rect(center=(GRID_WIDTH * BLOCK_SIZE // 2 + offset_x,
                                     GRID_HEIGHT * BLOCK_SIZE // 2))
        screen.blit(text, rect)

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Tetris - Gesture Control")
    clock = pygame.time.Clock()
    
    game = TetrisGame()
    try:
        gesture_controller = GestureController('models/best_prod_model_SVM (Linear).pkl')
    except Exception as e:
        print(f"Ошибка: {e}")
        sys.exit(1)
    
    running = True
    last_fall_time = pygame.time.get_ticks()
    
    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    game = TetrisGame()
        
        gesture, camera_frame = gesture_controller.get_gesture()
        
        if gesture == 'open_palm':
            game.paused = not game.paused
        
        # Движения только если не пауза и не game over
        if not game.paused and not game.game_over:
            if gesture == 'left':
                game.move_left()
            elif gesture == 'right':
                game.move_right()
            elif gesture == 'up':
                game.rotate_piece()
            elif gesture == 'down':
                game.move_down()
            
            # Автопадение
            if current_time - last_fall_time > game.fall_speed:
                game.move_down()
                last_fall_time = current_time
        
        draw_game(screen, game, camera_frame, gesture_controller)
        pygame.display.flip()
        clock.tick(FPS)
    
    gesture_controller.cap.release()
    pygame.quit()

if __name__ == "__main__":
    main() 