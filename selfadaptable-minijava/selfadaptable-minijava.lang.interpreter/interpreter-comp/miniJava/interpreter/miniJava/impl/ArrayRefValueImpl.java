package miniJava.interpreter.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import java.lang.String;
import miniJava.interpreter.miniJava.ArrayInstance;
import miniJava.interpreter.miniJava.ArrayRefValue;
import miniJava.interpreter.miniJava.MiniJavaFactory;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.Value;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecoretools.ale.compiler.lib.CollectionService;

public class ArrayRefValueImpl extends ValueImpl implements ArrayRefValue {
	protected ArrayInstance instance;

	protected ArrayRefValueImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.ARRAY_REF_VALUE;
	}

	public ArrayInstance getInstance() {
		if (instance != null && instance.eIsProxy()) {
			InternalEObject oldInstance = (InternalEObject) instance;
			instance = (ArrayInstance) eResolveProxy(oldInstance);
			if (instance != oldInstance) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, MiniJavaPackage.ARRAY_REF_VALUE__INSTANCE, oldInstance, instance));
			}
		}
		return instance;
	}

	public ArrayInstance basicGetInstance() {
		return instance;
	}

	public void setInstance(ArrayInstance newInstance) {
		ArrayInstance oldInstance = instance;
		instance = newInstance;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, MiniJavaPackage.ARRAY_REF_VALUE__INSTANCE, oldInstance, instance));
	}

	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case MiniJavaPackage.ARRAY_REF_VALUE__INSTANCE :
				if (resolve)
					return getInstance();
				return basicGetInstance();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case MiniJavaPackage.ARRAY_REF_VALUE__INSTANCE :
				setInstance((ArrayInstance) newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.ARRAY_REF_VALUE__INSTANCE :
				setInstance((ArrayInstance) null);
				return;
		}
		super.eUnset(featureID);
	}

	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case MiniJavaPackage.ARRAY_REF_VALUE__INSTANCE :
				return instance != null;
		}
		return super.eIsSet(featureID);
	}

	public Value copyj() {
		Value result;
		ArrayRefValue tmp = ((ArrayRefValue) (MiniJavaFactory.eINSTANCE.createArrayRefValue()));
		tmp.setInstance(this.getInstance());
		result = (Value) (tmp) ;
		return result;
	}

	public String customToString() {
		String result;
		String res = ((String) ("["));
		int i = ((int) (0));
		int lgt = ((int) (CollectionService.size(this.getInstance().getValue())));
		while ((i) < (lgt)) {
			Value tmpv = ((Value) (CollectionService.get(this.getInstance().getValue(), i)));
			res = (res) + (((Value) (tmpv)).customToString());
			if ((i) < ((lgt) - (1))) {
				res = (res) + (", ");
			}
			i = (i) + (1);
		}
		res = (res) + ("]");
		result = (String) (res) ;
		return result;
	}
}
