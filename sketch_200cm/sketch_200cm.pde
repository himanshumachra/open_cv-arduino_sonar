import processing.serial.*;
import processing.sound.*;

Serial myPort;
SoundFile hooter;
PrintWriter output; // <--- ADDED: To handle file writing

// --- STATE CONTROL ---
boolean isPlaying = false;      
boolean hasTriggered = false;   
boolean hooterHasFired = false; 
int detectionCounter = 0;       

// --- DATA ---
String angle = "";
String distance = "";
String data = "";
int iAngle = 0;
int iDistance = 0;
int index1;
float pixsDistance;
float maxDistance = 40.0; 

void setup() {
  size(1280, 720);
  smooth();
  
  // Create the log file in the sketch folder
  // "true" means it will overwrite/create a new file each time you start
  output = createWriter("radar_data_log.txt"); 
  output.println("Timestamp, Angle, Distance(cm)"); // Write a header

  PFont orcFont = createFont("SansSerif", 30);
  textFont(orcFont);

  try {
    myPort = new Serial(this, "COM3", 9600);
    myPort.bufferUntil('.');
  } catch (Exception e) {
    println("ERROR: COM3 not found.");
  }

  try {
    hooter = new SoundFile(this, "hooter.mp3");
  } catch (Exception e) {
    println("WARNING: hooter.mp3 not found.");
  }
}

void draw() {
  fill(0, 20); 
  noStroke();
  rect(0, 0, width, height);

  drawRadar();
  drawLine();
  drawObject();
  drawText();
}

void serialEvent(Serial myPort) {
  data = myPort.readStringUntil('.');
  if (data != null && data.length() > 1) {
    data = data.substring(0, data.length() - 1);
    index1 = data.indexOf(",");
    if (index1 > 0) {
      angle = data.substring(0, index1);
      distance = data.substring(index1 + 1);
      iAngle = int(angle);
      iDistance = int(distance);
      
      // --- LOG TO FILE ---
      // This saves the data to the text file every time a signal comes in
      output.println(hour() + ":" + minute() + ":" + second() + ", " + iAngle + ", " + iDistance);
      output.flush(); // Forces the data to be written to the file immediately
    }
  }
}

// Ensure the file saves properly when you close the program
void keyPressed() {
  if (key == 's' || key == 'S') {
    output.flush(); // Writes the remaining data to the file
    output.close(); // Finishes the file
    println("Log saved and closed.");
    exit(); // Stops the program
  }
}

void drawRadar() {
  pushMatrix();
  translate(width / 2, height - height * 0.074);
  noFill();
  stroke(98, 245, 31);
  strokeWeight(2);
  arc(0, 0, width * 0.9, width * 0.9, PI, TWO_PI);
  arc(0, 0, width * 0.675, width * 0.675, PI, TWO_PI);
  arc(0, 0, width * 0.45, width * 0.45, PI, TWO_PI);
  arc(0, 0, width * 0.225, width * 0.225, PI, TWO_PI);
  line(-width / 2, 0, width / 2, 0);
  for (int i = 30; i <= 150; i += 30) {
    line(0, 0, (-width / 2) * cos(radians(i)), (-width / 2) * sin(radians(i)));
  }
  popMatrix();
}

void drawObject() {
  pushMatrix();
  translate(width / 2, height - height * 0.074);
  stroke(255, 10, 10);
  strokeWeight(8);

  pixsDistance = map(iDistance, 0, maxDistance, 0, width * 0.45);

  if (iDistance > 0 && iDistance <= maxDistance) {
    line(pixsDistance * cos(radians(iAngle)), -pixsDistance * sin(radians(iAngle)), 
         (width * 0.45) * cos(radians(iAngle)), -(width * 0.45) * sin(radians(iAngle)));
    
    detectionCounter++;

    if (detectionCounter > 15 && !hasTriggered) {
      println("⚠️ PERIMETER BREACH: Executing Tactical Vision...");
      thread("launchPython"); 
      hasTriggered = true; 
    }
  } else if (iDistance > maxDistance + 10 || iDistance <= 0) {
    detectionCounter = 0;
  }
  popMatrix();
}

void launchPython() {
  String scriptPath = "C:\\Users\\himanshu\\Desktop\\prjct cv\\vision_system.py";
  String runCmd = "python \"" + scriptPath + "\"";
  String[] command = {"cmd", "/c", "start", "cmd", "/c", runCmd}; 
  
  try {
    exec(command);
    println("✅ Vision system launched.");
  } catch (Exception e) {
    println("Launch Failed: " + e.getMessage());
  }
}

void drawLine() {
  pushMatrix();
  stroke(30, 250, 60);
  strokeWeight(6);
  translate(width / 2, height - height * 0.074);
  line(0, 0, (width * 0.45) * cos(radians(iAngle)), -(width * 0.45) * sin(radians(iAngle)));
  popMatrix();
}

void drawText() {
  fill(0);
  noStroke();
  rect(0, height - 70, width, 70);

  fill(98, 245, 31);
  textSize(25);
  text("10 cm", width * 0.61, height - 20);
  text("20 cm", width * 0.72, height - 20);
  text("30 cm", width * 0.83, height - 20);
  text("40 cm", width * 0.93, height - 20);

  textSize(30);
  text("Angle: " + iAngle + " ", 50, height - 20);

  if (iDistance > 0 && iDistance <= maxDistance) {
    text("Distance: " + iDistance + " cm", 250, height - 20);
    fill(255, 0, 0);
    textSize(40);
    text("⚠ OBJECT DETECTED ⚠", width / 2 - 180, height / 2);

    if (hooter != null && !hooterHasFired) {
      hooter.play();           
      hooterHasFired = true;   
      println("🔊 Hooter beeped once. Locked for this session.");
    }
  } else {
    text("Out of Range", 250, height - 20);
  }
}
