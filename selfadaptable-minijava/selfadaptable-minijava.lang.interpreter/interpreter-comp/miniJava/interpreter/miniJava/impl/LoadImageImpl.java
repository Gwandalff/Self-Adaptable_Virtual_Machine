package miniJava.interpreter.miniJava.impl;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;

import miniJava.interpreter.miniJava.ArrayInstance;
import miniJava.interpreter.miniJava.ArrayRefValue;
import miniJava.interpreter.miniJava.Expression;
import miniJava.interpreter.miniJava.IntegerValue;
import miniJava.interpreter.miniJava.LoadImage;
import miniJava.interpreter.miniJava.MiniJavaFactory;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.State;
import miniJava.interpreter.miniJava.Value;

public class LoadImageImpl extends ExpressionImpl implements LoadImage {
	protected Expression path;

	protected LoadImageImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.LOAD_IMAGE;
	}

	public Expression getPath() {
		return path;
	}

	public NotificationChain basicSetPath(Expression newPath, NotificationChain msgs) {
		Expression oldPath = path;
		path = newPath;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.LOAD_IMAGE__PATH, oldPath, newPath);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setPath(Expression newPath) {
		if (newPath != path) {
			NotificationChain msgs = null;
			if (path != null)
				msgs = ((InternalEObject) path).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.LOAD_IMAGE__PATH, null, msgs);
			if (newPath != null)
				msgs = ((InternalEObject) newPath).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.LOAD_IMAGE__PATH, null, msgs);
			msgs = basicSetPath(newPath, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.LOAD_IMAGE__PATH, newPath, newPath));
	}

	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID,
			NotificationChain msgs) {
		switch (featureID) {
			case MiniJavaPackage.LOAD_IMAGE__PATH :
				return basicSetPath(null, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.LOAD_IMAGE__PATH :
				return getPath();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.LOAD_IMAGE__PATH :
				setPath((Expression) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.LOAD_IMAGE__PATH :
				setPath((Expression) null);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.LOAD_IMAGE__PATH :
				return path != null;
		}
		return super.eIsSet(featureID);
	}

	public Value evaluateExpression(State state) {
		Value result;
		result = (Value) (LoadImageImpl.loadImage((State) (state), (String) (((Value) (((Expression) (this.getPath())).evaluateExpression((State) (state)))).customToString()))) ;
		return result;
	}
	
	public static Value loadImage(State state, String path) {
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
		state.getArraysHeap().add((miniJava.interpreter.miniJava.ArrayInstance) res);
		
		for (int y = 0; y < h; y++) {
			ArrayInstance line = factory.createArrayInstance();
			line.setSize(w);
			for (int x = 0; x < w; x++) {
				IntegerValue idv = factory.createIntegerValue();
				int rgb = img.getRGB(x, y);
				int b = (rgb)&0xFF;
				int g = (rgb>>8)&0xFF;
				int r = (rgb>>16)&0xFF;
				idv.setValue((r+g+b)/3);
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
}
