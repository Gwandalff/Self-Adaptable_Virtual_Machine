package miniJava.interpreter.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.interpreter.miniJava.ClassRef;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.TypeDeclaration;
import miniJava.interpreter.miniJava.TypeRef;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;

public class ClassRefImpl extends SingleTypeRefImpl implements ClassRef {
	protected TypeDeclaration referencedClass;

	protected ClassRefImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.CLASS_REF;
	}

	public TypeDeclaration getReferencedClass() {
		if (referencedClass != null && referencedClass.eIsProxy()) {
			InternalEObject oldReferencedClass = (InternalEObject) referencedClass;
			referencedClass = (TypeDeclaration) eResolveProxy(oldReferencedClass);
			if (referencedClass != oldReferencedClass) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, MiniJavaPackage.CLASS_REF__REFERENCED_CLASS, oldReferencedClass, referencedClass));
			}
		}
		return referencedClass;
	}

	public TypeDeclaration basicGetReferencedClass() {
		return referencedClass;
	}

	public void setReferencedClass(TypeDeclaration newReferencedClass) {
		TypeDeclaration oldReferencedClass = referencedClass;
		referencedClass = newReferencedClass;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.CLASS_REF__REFERENCED_CLASS, oldReferencedClass, referencedClass));
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.CLASS_REF__REFERENCED_CLASS :
				if (resolve)
					return getReferencedClass();
				return basicGetReferencedClass();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.CLASS_REF__REFERENCED_CLASS :
				setReferencedClass((TypeDeclaration) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.CLASS_REF__REFERENCED_CLASS :
				setReferencedClass((TypeDeclaration) null);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.CLASS_REF__REFERENCED_CLASS :
				return referencedClass != null;
		}
		return super.eIsSet(featureID);
	}

	public boolean compare(TypeRef other) {
		boolean result;
		if (other instanceof ClassRef) {
			ClassRef ocr = ((ClassRef) (other));
			result = (boolean) (EqualService.equals((this.getReferencedClass()), (ocr.getReferencedClass()))) ;
		}
		else {
			result = (boolean) (false) ;
		}
		return result;
	}
}
