public enum Direction{

    DOWN(new Coordinate(1, 0)),
    RIGHT(new Coordinate(0, 1)),
    UP(new Coordinate(-1, 0)),
    LEFT(new Coordinate(0, -1)),
    NULL(new Coordinate(0, 0));

    public final Coordinate vector;

    Direction(Coordinate vector) {
        this.vector = vector;
    }

    // Method to check if two directions are opposite
    public boolean isOpposite(Direction other) {
        return this.vector.getX() + other.vector.getX() == 0 && this.vector.getY() + other.vector.getY() == 0;
    }

}


