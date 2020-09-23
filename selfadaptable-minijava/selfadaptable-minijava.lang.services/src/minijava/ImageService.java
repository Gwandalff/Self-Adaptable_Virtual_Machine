package minijava;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import miniJava.ArrayInstance;
import miniJava.ArrayRefValue;
import miniJava.IntegerValue;
import miniJava.MiniJavaFactory;
import miniJava.State;

public class ImageService {
	public static Object loadImage(Object self, Object state, String path) {
		BufferedImage img = null;
		try {
		    img = ImageIO.read(new File(path));
		} catch (IOException e) {
			System.out.println("ERROR loading image");
			return null;
		}
		int h = img.getHeight();
		int w = img.getWidth();
		
		MiniJavaFactory factory = MiniJavaFactory.eINSTANCE;
		
		ArrayInstance res = factory.createArrayInstance();
		res.setSize(h);
		//((State) state).getArraysHeap().add(res);
		
		for (int y = 0; y < h; y++) {
			ArrayInstance line = factory.createArrayInstance();
			line.setSize(w);
			for (int x = 0; x < w; x++) {
				IntegerValue idv = factory.createIntegerValue();
				idv.setValue(img.getRGB(x, y));
				line.getValue().add(idv);
			}
			ArrayRefValue lineRef = factory.createArrayRefValue();
			lineRef.setInstance(line);
			res.getValue().add(lineRef);
		}
		
		ArrayRefValue ret = factory.createArrayRefValue();
		ret.setInstance(res);
		return ret;
	}
	
	public static void writeImage(Object self, String path, Object image) {
		ArrayInstance imgInstance = ((ArrayRefValue) image).getInstance();
		int h = imgInstance.getSize();
		int w = ((ArrayRefValue) imgInstance.getValue().get(0)).getInstance().getSize();
		
		BufferedImage jImage = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
		for (int y = 0; y < h; y++) {
			ArrayInstance line = ((ArrayRefValue) imgInstance.getValue().get(y)).getInstance();
			for (int x = 0; x < w; x++) {
				IntegerValue idv = (IntegerValue) line.getValue().get(x);
				jImage.setRGB(x, y, idv.getValue());
			}
		}
		File output = new File(path);
		try {
			ImageIO.write(jImage, path.substring(path.lastIndexOf("."+1)), output);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}