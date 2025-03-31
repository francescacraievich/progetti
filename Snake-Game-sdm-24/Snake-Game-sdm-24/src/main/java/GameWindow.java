import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.plaf.basic.BasicButtonUI;
import java.awt.*;
import java.awt.event.ActionListener;
import java.awt.event.*;
import java.io.File;
import java.io.IOException;


public class GameWindow extends JFrame implements ActionListener{
    private final SnakeMovement snakeMovement;
    private Direction keyEvent;
    // Class extending JPanel for the game interface
    private final GamePanel gamePanel;

    // Jpanel for the menu interface
    JPanel menuPanel;

    // Buttons needed
    JButton newGameButton;
    JButton exitGameButton;
    JButton returnToMenuButton;

    public GameWindow(SnakeMovement snakeMovement) {
        this.snakeMovement = snakeMovement;
        //this.board = board;
        this.setTitle("Snake Game");
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); // Default exit operation
        this.setLayout(new BorderLayout());
        this.gamePanel = new GamePanel(snakeMovement); // snakeMovement works as controller
        this.menuPanel = setMenu();
        this.add(menuPanel);
        this.setVisible(true);
        this.setSize(650, 650);
        this.setResizable(true);
        this.setLocationRelativeTo(null);
    }

    public JPanel setMenu() {
        JPanel menuPanel = new JPanel();
        // Setting grid layout of 4 rows and 1 column
        menuPanel.setLayout(new BorderLayout());
        menuPanel.setBackground(Color.darkGray); // Set background color

        // Adding the title
        JLabel title = new JLabel("Snake", SwingConstants.CENTER);

        title.setForeground(Color.RED);

        try{
            Font customFont = Font.createFont(Font.TRUETYPE_FONT, new File("src/main/java/VT323-Regular.ttf")).deriveFont(Font.BOLD, 145f);
            title.setFont(customFont);
        }catch(FontFormatException | IOException e){
            title.setFont(new Font("Impact", Font.BOLD, 99));
            System.err.println("Custom font not loaded: " + e.getMessage());
        }//catch

        menuPanel.add(title, BorderLayout.CENTER);

        JPanel buttonPanel = new JPanel();
        buttonPanel.setLayout(new FlowLayout(FlowLayout.CENTER, 35, 130));
        buttonPanel.setBackground(Color.darkGray);

        newGameButton = createStyledButton("New Game");
        newGameButton.addActionListener(this);
        exitGameButton = createStyledButton("Exit Game");
        exitGameButton.addActionListener(this);

        //Adding buttons to panel
        buttonPanel.add(newGameButton);
        buttonPanel.add(exitGameButton);

        menuPanel.add(buttonPanel, BorderLayout.SOUTH);


        return menuPanel;
    }


    @Override
    public void actionPerformed(ActionEvent e){
        if(e.getSource() == newGameButton){
            GamePlay gameplay = new GamePlay(this);
            gameplay.start();
        } else if(e.getSource() == exitGameButton){
            this.dispose();
        } else {
            throw new IllegalStateException("Unexpected value: " + e.getSource());
        }
    }

    static class GamePanel extends JPanel {
        private final SnakeMovement snakeMovement;
        private Cell[][] board;


        public GamePanel(SnakeMovement snakeMovement){
            this.snakeMovement = snakeMovement;
            Cell[][] board = snakeMovement.getBoardState();
            this.setBoard(board);
        }// constructor

        public void setBoard(Cell[][] board) {
            this.board = board;
            repaint(); // Request the panel to be repainted
        }//setting up board

        @Override
        protected void paintComponent(Graphics g){
            if(board != null){

                int cellSize = 20;


                for (int i = 0; i < board.length;i++){
                    for(int j = 0; j < board[i].length; j++){
                        int x = j * cellSize; // Horizontal position of the cell
                        int y = i * cellSize; // Vertical position of the cell

                        switch(board[i][j]){
                            case BLANK:
                                g.setColor(Color.WHITE);
                                g.fillRect(x, y, cellSize, cellSize);
                                break;
                            case WALL:
                                g.setColor(Color.BLACK);
                                g.fillRect(x, y, cellSize, cellSize);
                                break;
                            case FOOD:
                                g.setColor(Color.RED);
                                g.fillOval(x, y, cellSize, cellSize);
                                break;
                            case SNAKE: //both body and head are green
                                g.setColor(Color.GREEN);
                                g.fillOval(x, y, cellSize, cellSize);
                                break;
                            default:
                                throw new RuntimeException("Draw game error out of bounds");
                        }//switch
                        g.fillOval(x, y, cellSize, cellSize);
                    }
                }
                String scoreText = "Score: " + snakeMovement.getScore();
                g.setColor(Color.WHITE);
                g.drawString(scoreText, 10, getHeight() - 5);
            }// if statement

        }

    }// GamePanel class

    class GamePlay extends Thread{
        private final JFrame frame;


        public GamePlay(JFrame frame){
            this.frame = frame;
        }

        @Override
        public void run(){
            // Removing the menu panel and adding the GamePanel to the window
            frame.remove(menuPanel);
            frame.add(gamePanel,BorderLayout.CENTER);
            frame.setFocusable(true);
            frame.setFocusTraversalKeysEnabled(false);
            frame.addKeyListener(new ArrowKeyListener()); // Adding KeyListener for arrow keys
            frame.setSize(616,639);
            keyEvent = Direction.NULL;

            Cell[][] board = snakeMovement.getBoardState();
            gamePanel.setBoard(board);

            SwingUtilities.invokeLater(() -> {
                frame.setVisible(true); // Making the frame visible
                frame.requestFocusInWindow(); // Request focus
            });

            // Main game loop!
            while(true){
                try{

                    int speed = snakeMovement.getSpeed();
                    int sleepTime = 300 - speed;

                    if(sleepTime < 100){
                        sleepTime = 100; // Minimum sleep -> Not quicker than this
                    }

                    Thread.sleep(sleepTime);
                } catch(InterruptedException e){
                    throw new RuntimeException(e);
                } // catch
                snakeMovement.moveSnake(keyEvent);
                if(snakeMovement.isGameOver()){
                    break;
                }
                board = snakeMovement.getBoardState();
                gamePanel.setBoard(board);

            } // end While

            frame.remove(gamePanel);
            JLabel finalOutput;
            Cell[][] boardSize = snakeMovement.getBoardState();
            int max_length = (boardSize.length - 2)*(boardSize.length - 2);
            if(snakeMovement.isGameOver()){
                finalOutput = new JLabel("Game Over!", SwingConstants.CENTER);
            } else if (snakeMovement.getSnake().getCoordSnake().size() >= max_length){
                finalOutput = new JLabel("You won!", SwingConstants.CENTER);
            } else{
                finalOutput = new JLabel("Error");
            }

            finalPanel(finalOutput);
        } // run method

        // FINAL PANEL
        public void finalPanel(JLabel label){
            JPanel finalPanel = new JPanel();
            finalPanel.setLayout(new BorderLayout());
            finalPanel.setBackground(Color.darkGray);
            label.setForeground(Color.RED);
            JPanel buttonPanel2 = new JPanel();
            buttonPanel2.setLayout(new FlowLayout(FlowLayout.CENTER, 35, 130));
            buttonPanel2.setBackground(Color.darkGray);

            returnToMenuButton = createStyledButton("BACK");
            returnToMenuButton.addActionListener(e -> {
                frame.dispose();
                Board b = new Board();
                SnakeMovement s = new SnakeMovement(b);
                new GameWindow(s);
            });

            try{
                Font customFont = Font.createFont(Font.TRUETYPE_FONT, new File("src/main/java/VT323-Regular.ttf")).deriveFont(Font.BOLD, 145f);
                label.setFont(customFont);
            }catch(FontFormatException | IOException e){
                label.setFont(new Font("Impact", Font.BOLD, 15));
                System.err.println("Custom font not loaded: " + e.getMessage());
            }//catch

            buttonPanel2.add(returnToMenuButton);
            finalPanel.add(label, BorderLayout.CENTER);
            finalPanel.add(buttonPanel2, BorderLayout.SOUTH);
            frame.add(finalPanel);
            frame.setSize(700, 700);
            frame.revalidate();
            frame.repaint();
        } //finalPanel

        class ArrowKeyListener extends KeyAdapter{
            @Override
            public void keyPressed(KeyEvent e){
                int key = e.getKeyCode();
                switch(key){
                    case (KeyEvent.VK_LEFT):
                        keyEvent = Direction.LEFT;
                        break;
                    case (KeyEvent.VK_RIGHT):
                        keyEvent = Direction.RIGHT;
                        break;
                    case (KeyEvent.VK_UP):
                        keyEvent = Direction.UP;
                        break;
                    case (KeyEvent.VK_DOWN):
                        keyEvent = Direction.DOWN;
                        break;
                }
            }
        }//ArrowKeyListener
    }//GamePlay class


    private JButton createStyledButton(String text) {
        JButton button = new JButton(text);
        button.setFont(new Font("Calibri", Font.BOLD, 17)); // Set font
        button.setForeground(Color.WHITE); // Set text color
        button.setBackground(new Color(0x2dce98)); // Set background color
        button.setFocusPainted(false); // Remove focus border
        button.setPreferredSize(new Dimension(200, 70));
        button.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR)); // Change cursor on hover
        button.setUI(new StyledButton());
        return button;
    }

    // Styling the button with STYLED BUTTON class
    class StyledButton extends BasicButtonUI{
        @Override
        public void installUI (JComponent c) {
            super.installUI(c);
            AbstractButton button = (AbstractButton) c;
            button.setOpaque(false);
            button.setBorder(new EmptyBorder(5, 15, 5, 15));

        }

        @Override
        public void paint (Graphics g, JComponent c) {
            AbstractButton b = (AbstractButton) c;
            paintBackground(g, b, b.getModel().isPressed() ? 2 : 0);
            super.paint(g, c);
        }

        private void paintBackground (Graphics g, JComponent c, int yOffset) {
            Dimension size = c.getSize();
            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g.setColor(c.getBackground().darker());
            g.fillRoundRect(0, yOffset, size.width, size.height - yOffset, 20, 20);
            g.setColor(c.getBackground());
            g.fillRoundRect(0, yOffset, size.width, size.height + yOffset - 10, 20, 20);
        }

    }// StyledButton

}// GameWindow class
