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
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.State;
import miniJava.interpreter.miniJava.Value;
import miniJava.interpreter.miniJava.WriteImage;
import minijava.ImageService;

public class WriteImageImpl extends StatementImpl implements WriteImage {
	protected Expression path;

	protected Expression image;

	protected WriteImageImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.WRITE_IMAGE;
	}

	public Expression getPath() {
		return path;
	}

	public NotificationChain basicSetPath(Expression newPath, NotificationChain msgs) {
		Expression oldPath = path;
		path = newPath;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.WRITE_IMAGE__PATH, oldPath, newPath);
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
				msgs = ((InternalEObject) path).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.WRITE_IMAGE__PATH, null, msgs);
			if (newPath != null)
				msgs = ((InternalEObject) newPath).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.WRITE_IMAGE__PATH, null, msgs);
			msgs = basicSetPath(newPath, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.WRITE_IMAGE__PATH, newPath, newPath));
	}

	public Expression getImage() {
		return image;
	}

	public NotificationChain basicSetImage(Expression newImage, NotificationChain msgs) {
		Expression oldImage = image;
		image = newImage;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, MiniJavaPackage.WRITE_IMAGE__IMAGE, oldImage, newImage);
			if (msgs == null)
				msgs = notification;
			else
				msgs.add(notification);
		}
		return msgs;
	}

	public void setImage(Expression newImage) {
		if (newImage != image) {
			NotificationChain msgs = null;
			if (image != null)
				msgs = ((InternalEObject) image).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.WRITE_IMAGE__IMAGE, null, msgs);
			if (newImage != null)
				msgs = ((InternalEObject) newImage).eInverseAdd(this, EOPPOSITE_FEATURE_BASE - MiniJavaPackage.WRITE_IMAGE__IMAGE, null, msgs);
			msgs = basicSetImage(newImage, msgs);
			if (msgs != null)
				msgs.dispatch();
		} else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.WRITE_IMAGE__IMAGE, newImage, newImage));
	}

	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID,
			NotificationChain msgs) {
		switch (featureID) {
			case MiniJavaPackage.WRITE_IMAGE__PATH :
				return basicSetPath(null, msgs);
			case MiniJavaPackage.WRITE_IMAGE__IMAGE :
				return basicSetImage(null, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.WRITE_IMAGE__PATH :
				return getPath();
			case MiniJavaPackage.WRITE_IMAGE__IMAGE :
				return getImage();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.WRITE_IMAGE__PATH :
				setPath((Expression) newValue);
				return;
			case MiniJavaPackage.WRITE_IMAGE__IMAGE :
				setImage((Expression) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.WRITE_IMAGE__PATH :
				setPath((Expression) null);
				return;
			case MiniJavaPackage.WRITE_IMAGE__IMAGE :
				setImage((Expression) null);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.WRITE_IMAGE__PATH :
				return path != null;
			case MiniJavaPackage.WRITE_IMAGE__IMAGE :
				return image != null;
		}
		return super.eIsSet(featureID);
	}

	public void evaluateStatement(State state) {
		WriteImageImpl.writeImage((String) (((Value) (((Expression) (this.getPath())).evaluateExpression((State) (state)))).customToString()), (Value) (((Expression) (this.getImage())).evaluateExpression((State) (state))));
	}
	
	public static void writeImage(String path, Object image) {
		ArrayInstance imgInstance = ((ArrayRefValue) image).getInstance();
		int h = imgInstance.getSize();
		int w = ((ArrayRefValue) imgInstance.getValue().get(0)).getInstance().getSize();
		
		BufferedImage jImage = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
		for (int y = 0; y < h; y++) {
			ArrayInstance line = ((ArrayRefValue) imgInstance.getValue().get(y)).getInstance();
			for (int x = 0; x < w; x++) {
				IntegerValue idv = (IntegerValue) line.getValue().get(x);
				int grey = idv.getValue();
				jImage.setRGB(x, y, grey+grey*256+grey*256*256);
			}
		}
		File output = new File(path);
		try {
			ImageIO.write(jImage, path.substring(path.lastIndexOf(".")+1), output);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
