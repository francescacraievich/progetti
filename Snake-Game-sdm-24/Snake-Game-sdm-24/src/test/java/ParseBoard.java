import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class ParseBoard {

    @Test
    void boardSizeIsThirty() {
        assertEquals(30, Board.getBoardSize());
    }

    @Test
    @Disabled
    void firstCellHasSnakeHead() {
        Board board = new Board();
        assertEquals(Cell.SNAKE, board.getCell(0, 0));
    }

    @Test
    @Disabled
    void secondCellHasFood() {
        Board board = new Board();
        assertEquals(Cell.FOOD, board.getCell(0, 1));
    }

    @Test
    void boardHasSnakeLongOne() {
        Board board = new Board();
        int snakeLength = 0;
        for (int i = 0; i < Board.getBoardSize(); i++) {
            for (int j = 0; j < Board.getBoardSize(); j++) {
                if (board.getCell(i, j) == Cell.SNAKE) {
                    snakeLength++;
                }
            }
        }
        assertEquals(1, snakeLength);
    }

    @Test
    void boardHasOneFood() {
        Board board = new Board();
        int foodCounter = 0;
        for (int i = 0; i < Board.getBoardSize(); i++) {
            for (int j = 0; j < Board.getBoardSize(); j++) {
                if (board.getCell(i, j) == Cell.FOOD) {
                    foodCounter++;
                }
            }
        }
        assertEquals(1, foodCounter);
    }

    @Test
    void foodAdditionInCenterCell() {
        Board board = new Board();
        board.setCell(4, 4, Cell.FOOD);
        assertEquals(Cell.FOOD, board.getCell(4, 4));
    }

    @Test
    void wallsAtEdgesOfBoard() {
        Board board = new Board();
        for (int i = 0; i < Board.getBoardSize(); i++) {
            for (int j = 0; j < Board.getBoardSize(); j++) {
                if (i == 0 || i == Board.getBoardSize()-1 || j == 0 || j == Board.getBoardSize()-1) {
                    assertEquals(Cell.WALL, board.getCell(i, j));
                } else {
                    assertNotEquals(Cell.WALL, board.getCell(i, j));
                }
            }
        }
    }




    @Test
    @Disabled
    void newFoodIsSpawnedAfterEating() {
        Board board = new Board();
        Coordinate initialFoodPosition = board.getCoordinateFood(); //initial food position
        SnakeMovement snakeMovement = new SnakeMovement(board);

        snakeMovement.eatFood(initialFoodPosition);
        board.regenerateFood(); // new piece of food

        Coordinate newFoodPosition = board.getCoordinateFood(); //new position of the food

        assertNotEquals(initialFoodPosition, newFoodPosition); //new food in a different position
        assertEquals(Cell.FOOD, board.getCell(newFoodPosition.getX(), newFoodPosition.getY())); //the new position contains the food
    }


}
