package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.Expression;
import miniJava.visitor.miniJava.LoadImage;
import miniJava.visitor.miniJava.MiniJavaPackage;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import visitor.VisitorInterface;

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

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__LoadImage(this);
	}
}
