import java.util.ArrayList;

public class SnakeMovement {

    private final Board board;
    private Snake snake;
    private int score;
    private Direction currentDirection = Direction.NULL;

    // CoordBody[0] refers to the parte of the body closest to the head

    public SnakeMovement(Board board) {
        // This constructor instantiate a Snake object with only the head
        // If we need a snake with more than just the head we should modify it with snake.setCoordSnake(ArrayList<Coordinates>)
        this.board = board;
        this.snake = new Snake(board.getInitialCoordinateHead());
        this.score = 0;
    }

    public void moveSnake(Direction newDirection) {

        // Declaration of the variables used
        ArrayList<Coordinate> coordSnake = snake.getCoordSnake();  // get current snake coord
        Coordinate coordSnakeTail = coordSnake.get(coordSnake.size() - 1);  // get current tail coord
        Coordinate coordSnakeHeadBefore = coordSnake.get(0);  // get current hea

        // Check that newDirection is legal
        newDirection = checkIfNewDirectionIsLegal(newDirection, coordSnake);

        // Update the current direction with the newDirection
        this.currentDirection = newDirection;

        // Calculate head position after the move
        Coordinate coordSnakeHeadAfter = coordSnakeHeadBefore.plus(newDirection.vector);

        // Update the old head on the borad and add the new head to the snake ArrayList in position 0
        board.setCell(coordSnakeHeadBefore, Cell.SNAKE); // Current head becomes body
        coordSnake.add(0, coordSnakeHeadAfter); // Add new head

        // Determine if snake head moved over a food Cell (if coordSnakeHeadAfter == coordFood)
        boolean grows = coordSnakeHeadAfter.equals(board.getCoordinateFood());

        if (!grows) {
            // Remove tail from ArrayList and set it as BLANK
            coordSnake.remove(coordSnake.size() - 1);
            board.setCell(coordSnakeTail, Cell.BLANK); // Clear old tail
        } else {
            // Consume food and regenerate new food
            eatFood(coordSnakeHeadAfter);
            score++;
        }

        // Update the new head on the board (if grows, it replaces the food)
        board.setCell(coordSnakeHeadAfter, Cell.SNAKE);

        // Update snake Arraylist values
        snake.setCoordSnake(coordSnake);

    }

    // Checks if the newDirection is legal based on the size of the snake (size=1 can go anywhere, size>1 cannot turn 180° in one command)
    private Direction checkIfNewDirectionIsLegal(Direction newDirection, ArrayList<Coordinate> coordSnake) {
        if (newDirection.isOpposite(this.currentDirection) && coordSnake.size() > 1) {  // can't turn 180°
            newDirection = this.currentDirection;
        }
        return newDirection;
    }

    public boolean isGameOver() {
        ArrayList<Coordinate> coordSnake = snake.getCoordSnake();
        Coordinate head = coordSnake.get(0);  // The head is at index 0 in the list
        int headRow = head.getY();
        int headCol = head.getX();

        // Case 1: Check if the head is colliding with the wall
        if (headRow <= 0 || headRow >= board.getBoardSize() - 1 || headCol <= 0 || headCol >= board.getBoardSize() - 1) {
            return true;  // The game is over if the head goes out of bounds (hits the wall)
        }

        // second case : self collision
        ArrayList<Coordinate> coordBody = new ArrayList<>(coordSnake.subList(1, coordSnake.size())); // body without head

        System.out.println("head" + head + " body" + coordBody);

        if (coordBody.contains(head)) {
            System.out.println("Game Over: Head collided with body");
            return true;
        }

        return false;
    }


    public void eatFood(Coordinate foodPosition) {
        Coordinate head = snake.getCoordSnake().get(0);

        //if head is in the food position
        if (head.equals(foodPosition)) {
            //new food
            board.regenerateFood();
        }
    }


    public Direction getCurrentDirection() { //not used
        return currentDirection;
    }

    public void setCurrentDirection(Direction direction) {
        this.currentDirection = direction;
    }

    public Cell[][] getBoardState() {
        return this.board.getBoard();
    }

    public Snake getSnake() {
        return snake;
    }

    public void setSnake(Snake snake) {
        this.snake = snake;
    }

    public int getScore() {
        return this.score;
    }

    public int getSpeed(){
        return (int) (Math.log(score + 1) * 50);
    }


}

