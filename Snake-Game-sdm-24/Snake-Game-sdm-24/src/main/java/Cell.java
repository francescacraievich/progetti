public enum Cell {

    BLANK(0),
    SNAKE(1),
    FOOD(2),
    WALL(3);

    private final int value;

    Cell(int value){
        this.value = value;
    }
}
